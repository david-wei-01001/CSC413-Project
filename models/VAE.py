import torch
import torch.nn as nn





def diag_gaussian_samples(mean, log_std, num_samples):
    # mean and log_std are (D) dimensional vectors
    # Return a (num_samples, D) matrix, where each sample is
    # from a diagonal multivariate Gaussian.

    # REMOVE SOLUTION
    return mean + torch.exp(log_std) * torch.randn(num_samples, mean.shape[-1])


def calculate_KL(mean, log_std):
    return torch.sum(0.5 * (mean.pow(2) + log_std.exp().pow(2) - 1) - log_std)


class CNN(nn.Module):
    def __init__(self, matrixSize=32):
        super(CNN, self).__init__()

        self.convs = nn.Sequential(nn.Conv2d(512, 256, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(256, 128, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(128, matrixSize, 3, 1, 1))

        # 32x8x8
        self.fc = nn.Linear(matrixSize*matrixSize, matrixSize*matrixSize)

    def forward(self, x):
        out = self.convs(x)
        # 32x8x8
        b, c, h, w = out.size()
        out = out.view(b, c, -1)
        # 32x64
        out = torch.bmm(out, out.transpose(1, 2)).div(h*w)
        # 32x32
        out = out.view(out.size(0), -1)
        return self.fc(out)


class VAE(nn.Module):
    def __init__(self, latent_dimension):
        super(VAE, self).__init__()

        # 32x8x8
        self.a_encode = nn.Sequential(nn.Linear(512, 2 * latent_dimension),
                                    )
        self.bn = nn.BatchNorm1d(128)
        self.decode = nn.Sequential(nn.Linear(latent_dimension, 256),
                                    nn.BatchNorm1d(256),
                                    nn.ReLU(),
                                    nn.Linear(256, 512),
                                    )

    def reparameterize(self, k, a_mu, a_log_std, b_mu, b_log_std, batch_size):
        a_mu = self.bn(a_mu)
        b_mu = self.bn(b_mu)
        a_samples = diag_gaussian_samples(a_mu, a_log_std, batch_size)
        b_samples = diag_gaussian_samples(b_mu, b_log_std, batch_size)
        return k * a_samples + (1 - k) * b_samples

    def forward(self, k, a, b, cap=0.):
        # 32x8x8
        batch, cl, h = a.size()
        a = a.view(batch, -1)
        b = b.view(batch, -1)

        x = torch.cat((a, b), dim=0)
        mu, log_std = self.encode(x).chunk(2, dim=1)
        a_mu, b_mu = mu.chunk(2, dim=0)
        a_log_std, b_log_std = log_std.chunk(2, dim=0)
        if cap != 0.:
            k = min(cap, k)

        # reparameterize
        z_q = self.reparameterize(k, a_mu, a_log_std, b_mu, b_log_std, batch)
        out = self.decode(z_q)
        out = out.view(batch, cl, h)

        KL = k * calculate_KL(a_mu, a_log_std) + (1 - k) * calculate_KL(b_mu, b_log_std)

        return out, KL


class MulLayer(nn.Module):
    def __init__(self, latent_dimension, matrixSize=32):
        super(MulLayer, self).__init__()
        self.snet = CNN(matrixSize)
        self.VAE = VAE(latent_dimension=latent_dimension)
        self.cnet = CNN(matrixSize)
        self.matrixSize = matrixSize

        self.compress = nn.Conv2d(512, matrixSize, 1, 1, 0)
        self.unzip = nn.Conv2d(matrixSize, 512, 1, 1, 0)

        self.transmatrix = None

    def forward(self, k, cF, sF_1, sF_2):
        cb, cc, ch, cw = cF.size()
        cFF = cF.view(cb, cc, -1)
        cMean = torch.mean(cFF, dim=2, keepdim=True)
        cMean = cMean.unsqueeze(3)
        cMean = cMean.expand_as(cF)
        cF = cF - cMean

        sb, sc, sh, sw = sF_1.size()
        sFF_1 = sF_1.view(sb, sc, -1)
        sMean_1 = torch.mean(sFF_1, dim=2, keepdim=True)

        sFF_2 = sF_2.view(sb, sc, -1)
        sMean_2 = torch.mean(sFF_2, dim=2, keepdim=True)

        sMean, KL = self.VAE(k, sMean_1, sMean_2)
        sMean = sMean.unsqueeze(3)
        sF_1 = sF_1 - sMean.expand_as(sF_1)
        sF_2 = sF_2 - sMean.expand_as(sF_2)


        sMeanC = sMean.expand_as(cF)

        compress_content = self.compress(cF)
        b, c, h, w = compress_content.size()
        compress_content = compress_content.view(b, c, -1)

        cMatrix = self.cnet(cF)
        sMatrix = self.snet(k * sF_1 + (1 - k)*sF_2)

        sMatrix = sMatrix.view(sMatrix.size(0), self.matrixSize, self.matrixSize)
        cMatrix = cMatrix.view(cMatrix.size(0), self.matrixSize, self.matrixSize)
        transmatrix = torch.bmm(sMatrix, cMatrix)
        transfeature = torch.bmm(transmatrix, compress_content).view(b, c, h, w)
        out = self.unzip(transfeature.view(b, c, h, w))
        out = out + sMeanC
        return out, transmatrix, KL




