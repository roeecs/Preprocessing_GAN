from constants import *
from plot import *
from torchvision import transforms
from h5dataset import H5Dataset
import torch
from torch import autograd
from lc_help_funcs import create_random_labels, to_one_hot, gen_rand_noise
from architecture import Generator, Discriminator, weights_init
from tensorboardX import SummaryWriter
import time
from timeit import default_timer as timer
import torchvision

#  train constants - differ from general constants
fixed_noise = gen_rand_noise()
fixed_labels = create_random_labels(BATCH_SIZE, NUM_CLASSES)
if RESTORE_MODE:
    aG = torch.load(OUTPUT_PATH + "generator.pt")
    aD = torch.load(OUTPUT_PATH + "discriminator.pt")
else:
    aG = Generator(DIM, 64 * 64 * 3, NUM_CLASSES)
    aD = Discriminator(DIM, NUM_CLASSES)
    aG.apply(weights_init)
    aD.apply(weights_init)


def decay(iter_run):
    return max(0., 1. - (iter_run / END_ITER))


opt_g = torch.optim.Adam(aG.parameters(), lr=LR, betas=(0, 0.9))
opt_d = torch.optim.Adam(aD.parameters(), lr=LR, betas=(0, 0.9))
optimizer_g = torch.optim.lr_scheduler.LambdaLR(opt_g, decay, last_epoch=-1)
optimizer_d = torch.optim.lr_scheduler.LambdaLR(opt_d, decay, last_epoch=-1)
one = torch.FloatTensor([1])
mone = one * -1
aG = aG.to(device)
aD = aD.to(device)
one = one.to(device)
mone = mone.to(device)
writer = SummaryWriter()


def get_data_loader(path_to_file):
    """
    :param path_to_file: path to hdf5 containing
    :return: a data loader
    """

    data_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(64),
        transforms.CenterCrop(64),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    dataset = H5Dataset(path_to_file, transform=data_transform)
    dataset_loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=1,
                                                 drop_last=True, pin_memory=True)
    return dataset_loader


def training_data_loader():
    return get_data_loader(DATA_PATH)


def val_data_loader():
    return get_data_loader(VAL_PATH)


def calc_gradient_penalty(netD, real_data, fake_data, labels):
    """
    :param netD: network's discriminator
    :param real_data: batch size of real data images
    :param fake_data: batch size of generated images
    :param labels: labels acording to the real data
    :return: the calculated gradient penalty
    """
    alpha = torch.rand(BATCH_SIZE, 1)
    alpha = alpha.expand(BATCH_SIZE, int(real_data.nelement() / BATCH_SIZE)).contiguous()
    alpha = alpha.view(BATCH_SIZE, 3, DIM, DIM)
    alpha = alpha.to(device)

    fake_data = fake_data.view(BATCH_SIZE, 3, DIM, DIM)
    interpolates = alpha * real_data.detach() + ((1 - alpha) * fake_data.detach())

    interpolates = interpolates.to(device)
    interpolates.requires_grad_(True)
    disc_interpolates = netD(interpolates, labels)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
    return gradient_penalty


def generate_image(netG, noise=None):
    """
    :param netG: network's generator
    :param noise: a random noise
    :return: generated image
    """
    if noise is None:
        noise = gen_rand_noise()

    with torch.no_grad():
        noisev = noise
    samples = netG(noisev, fixed_labels)
    samples = samples.view(BATCH_SIZE, 3, 64, 64)
    samples = samples * 0.5 + 0.5
    return samples





def train():
    """
    main train function
    """
    dataloader = training_data_loader()
    dataiter = iter(dataloader)
    for iteration in range(START_ITER, END_ITER):
        start_time = time.time()
        print("Iter: " + str(iteration))
        start = timer()
        # ---------------------TRAIN G------------------------
        for p in aD.parameters():
            p.requires_grad_(False)  # freeze D

        gen_cost = None
        for i in range(GENER_ITERS):  # gener_iters=1
            print("Generator iters: " + str(i))
            aG.zero_grad()
            noise = gen_rand_noise()
            noise.requires_grad_(True)
            g_train_labels = create_random_labels(BATCH_SIZE, NUM_CLASSES)
            fake_data = aG(noise, g_train_labels)
            gen_cost = aD(fake_data, g_train_labels)
            gen_cost = gen_cost.mean()
            gen_cost.backward(mone)
            gen_cost = -gen_cost

        optimizer_g.step()
        end = timer()
        print(f'---train G elapsed time: {end - start}')
        # ---------------------TRAIN D------------------------
        for p in aD.parameters():  # reset requires_grad
            p.requires_grad_(True)  # they are set to False below in training G
        for i in range(CRITIC_ITERS):
            print("Critic iter: " + str(i))
            # gen fake data and load real data
            start = timer()

            batch_img, batch_labels = next(dataiter, None)
            if batch_img is None or batch_labels is None:
                dataiter = iter(dataloader)
                batch_img, batch_labels = dataiter.next()
            real_data = batch_img.to(device)
            batch_labels = batch_labels.to(device)
            batch_labels = to_one_hot(batch_labels, NUM_CLASSES)
            end = timer();
            print(f'---load real imgs elapsed time: {end-start}')

            start = timer()
            aD.zero_grad()
            noise = gen_rand_noise()
            with torch.no_grad():
                noisev = noise  # totally freeze G, training D
            fake_data = aG(noisev, batch_labels).detach()
            end = timer();
            print(f'---gen G elapsed time: {end-start}')

            start = timer()
            # train with real data
            disc_real = aD(real_data, batch_labels)
            disc_real = disc_real.mean()

            # train with fake data
            disc_fake = aD(fake_data, batch_labels)
            disc_fake = disc_fake.mean()

            # train with interpolates data
            gradient_penalty = calc_gradient_penalty(aD, real_data, fake_data, batch_labels)

            # final disc cost
            disc_cost = disc_fake - disc_real + gradient_penalty
            disc_cost.backward()
            w_dist = disc_fake - disc_real
            optimizer_d.step()

            # ------------------VISUALIZATION----------
            if i == CRITIC_ITERS - 1:
                writer.add_scalar('data/disc_cost', disc_cost, iteration)
                # writer.add_scalar('data/disc_fake', disc_fake, iteration)
                # writer.add_scalar('data/disc_real', disc_real, iteration)
                writer.add_scalar('data/gradient_pen', gradient_penalty, iteration)
                # writer.add_scalar('data/d_conv_weight_mean', [i for i in aD.children()][0].conv.weight.data.clone().mean(), iteration)
                # writer.add_scalar('data/d_linear_weight_mean', [i for i in aD.children()][-1].weight.data.clone().mean(), iteration)
                # writer.add_scalar('data/fake_data_mean', fake_data.mean())
                # writer.add_scalar('data/real_data_mean', real_data.mean())
                # if iteration %200==99:
                #    paramsD = aD.named_parameters()
                #    for name, pD in paramsD:
                #        writer.add_histogram("D." + name, pD.clone().data.cpu().numpy(), iteration)
            #                 if (iteration%2)==1:
            #                   body_model = [i for i in aD.children()][0]
            #                   print(body_model)
            #                   print(type(body_model))
            #                   layer1 = body_model.conv
            #                   print(layer1)
            #                   print(type(layer1))
            #                   xyz = layer1.weight.data.clone()
            #                   print(xyz)
            #                   print(xyz.shape)
            #                   print(type(xyz))
            #                   tensor = xyz.cpu()
            #                   tensors = torchvision.utils.make_grid(tensor, nrow=8,padding=1)
            #                   writer.add_image('D/conv1', tensors, iteration)

            end = timer()
            print(f'---train D elapsed time: {end-start}')

        # ---------------VISUALIZATION---------------------
        writer.add_scalar('data/gen_cost', gen_cost, iteration)
        plot(OUTPUT_PATH + 'time', time.time() - start_time)
        plot(OUTPUT_PATH + 'train_disc_cost', disc_cost.cpu().data.numpy())
        plot(OUTPUT_PATH + 'train_gen_cost', gen_cost.cpu().data.numpy())
        plot(OUTPUT_PATH + 'wasserstein_distance', w_dist.cpu().data.numpy())
        if (iteration % 10) == 9:
            val_loader = val_data_loader()
            val_dataiter = iter(val_loader)
            dev_disc_costs = []
            for i in range(1):
                batch_img, batch_labels = next(val_dataiter, None)
                if batch_img is None or batch_labels is None:
                    dataiter = iter(dataloader)
                    batch_img, batch_labels = dataiter.next()
                val_data = batch_img.to(device)
                batch_labels = batch_labels.to(device)
                batch_labels = to_one_hot(batch_labels, NUM_CLASSES)
                D = aD(val_data, batch_labels)
                _dev_disc_cost = -D.mean().cpu().data.numpy()
                dev_disc_costs.append(_dev_disc_cost)
            plot(OUTPUT_PATH + 'dev_disc_cost.png', np.mean(dev_disc_costs))
            flush()
            gen_images = generate_image(aG, fixed_noise)
            torchvision.utils.save_image(gen_images, OUTPUT_PATH + 'samples_{}.png'.format(iteration), nrow=8,
                                         padding=2)
            grid_images = torchvision.utils.make_grid(gen_images, nrow=8, padding=2)
            writer.add_image('images', grid_images, iteration)

            # ----------------------Save model----------------------
            torch.save(aG, OUTPUT_PATH + "generator.pt")
            torch.save(aD, OUTPUT_PATH + "discriminator.pt")
        tick()


if __name__ == '__main__':
    train()