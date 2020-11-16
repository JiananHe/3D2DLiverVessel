import numpy as np
from cv2 import GaussianBlur
import matplotlib.image as mpimg  # mpimg 用于读取图片



PROCESS_ROOT = 'process/'


def vesselness2D(I, sigmas, spacing, tau, brightondark):
    '''
    sigmas: 输入图像间距分辨率-在海森矩阵计算过程中，可以调整每个维的高斯滤波核大小，以适应不同维的图像间距
    tau: 参数τ
    brightondark(true/false): 是否为亮线暗背景

    调用：imageEigenvalues
    '''
    for j in range(len(sigmas)):
        print('Currtent filter scale (sigma):', sigmas[j])
        _, Lambda2 = imageEigenvalues(I, sigmas[j], spacing, brightondark)
        if (brightondark):
            Lambda2 = -Lambda2

        Lambda3 = np.copy(Lambda2)

        # 分段计算λρ
        Lambda_rho = np.copy(Lambda3)
        # pos = (Lambda3 > 0) & (Lambda3 <= tau * max(Lambda3))
        # The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()
        # 我恨这个错误
        # a = tau * Lambda3.max()
        pos = np.logical_and(Lambda3 > 0, Lambda3 <= tau * Lambda3.max())
        # Lambda_rho[((Lambda3 > 0) & Lambda3 <= tau * max(Lambda3))] = tau * max(Lambda3)
        Lambda_rho[pos] = tau * Lambda3.max()
        Lambda_rho[Lambda3 <= 0] = 0
        # # debug
        # plt.figure('Lambda_rho')
        # plt.imshow(Lambda_rho)
        # plt.show()
        mpimg.imsave(PROCESS_ROOT + 'Lambda_rho_sigma=%.1f' % sigmas[j] + 'tau=%.1f.png' % tau, Lambda_rho)

        # 增强函数结果
        response = np.power(Lambda2, 2) * (Lambda_rho - Lambda2) * 27 / np.power(Lambda2 + Lambda_rho, 3)
        pos = np.logical_and(Lambda2 >= Lambda_rho / 2, Lambda_rho > 0)
        response[pos] = 1
        # response[Lambda2 >= Lambda_rho/2 & Lambda_rho > 0] = 1
        response[(Lambda2 <= 0) | (Lambda_rho <= 0)] = 0
        # response[~isfinite(response)] = 0;
        mpimg.imsave(PROCESS_ROOT + 'response_sigma=%.1f' % sigmas[j] + 'tau=%.1f.png' % tau, response)
        if (j == 0):
            vesselness = response
        else:
            # vesselness = vesselness.astype(np.bool)
            # response = response.astype(np.bool)
            vesselness = np.max([vesselness, response], axis=0)
            # # debug
            # plt.figure('vesselness')
            # plt.imshow(vesselness)
            # plt.show()
        # 【待完成】
        mpimg.imsave(PROCESS_ROOT + 'vesselness_sigma=%.1f' % sigmas[j] + 'tau=%.1f.png' % tau, vesselness)
    vesselness = vesselness / vesselness.max()  # should not be really needed
    vesselness[vesselness < 1e-2] = 0
    return vesselness


def imageEigenvalues(I, sigma, spacing, brightondark):
    '''返回2d图像海森矩阵2个特征值图像'''
    # 计算二维海森矩阵
    Hxx, Hyy, Hxy = Hessian2D(I, sigma, spacing)
    # 保存图像
    mpimg.imsave(PROCESS_ROOT + '\Hxx_sigma=%.1f.png' % sigma, Hxx)
    mpimg.imsave(PROCESS_ROOT + '\Hxy_sigma=%.1f.png' % sigma, Hxy)
    mpimg.imsave(PROCESS_ROOT + '\Hyy_sigma=%.1f.png' % sigma, Hyy)

    # 尺度校正【不知道这一步是矫正滤波带来的误差还是求导带来的误差，
    # 如果是滤波的误差，我使用内置高斯函数会不会出问题？
    # 不过应该是求导引入的误差。因为仅仅卷积高斯核好像从来不需要校正。】
    c = pow(sigma, 2)
    Hxx = c * Hxx
    Hxy = c * Hxy
    Hyy = c * Hyy

    # 减小计算量的操作【这些论文里都没写。但代码有提到来自哪里】
    B1 = - (Hxx + Hyy);
    B2 = np.multiply(Hxx, Hyy) - np.power(Hxy, 2)
    T = np.ones(B1.shape)

    if (brightondark == True):
        T[B1 < 0] = 0
        # T[B2 == 0 and B1 == 0] = 0
        # The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()
        # T = np.where(B1 < 0 or (B2 == 0 and B1 == 0), 0, 1)
        # indeces = np.where(B2==0) and np.where(B1==0)
        # T[indeces] = 0
        T[(B2 == 0) & (B1 == 0)] = 0
        pass
    else:
        T[B1 > 0] = 0
        # indeces = (B2==0) and (B1==0)
        # T[(B2==0) & (B1==0)] = 0
        # T = np.where(B1 > 0 or (B2 == 0 and B1 == 0), 0, 1)
        T[(B2 == 0) & (B1 == 0)] = 0
        pass

    indeces = np.where(T == 1)
    Hxx = Hxx[indeces]
    Hyy = Hyy[indeces]
    Hxy = Hxy[indeces]

    # 只计算符合条件的位置的特征值
    Lambda1i, Lambda2i = eigvalOfHessian2D(Hxx, Hxy, Hyy)
    Lambda1 = np.zeros(T.shape)  # 这么说，其它位置特征值都是0
    Lambda2 = np.zeros(T.shape)
    Lambda1[indeces] = Lambda1i
    Lambda2[indeces] = Lambda2i
    # 去噪声
    Lambda1[np.abs(Lambda1) < 1e-4] = 0
    Lambda2[np.abs(Lambda2) < 1e-4] = 0
    # # debug
    # plt.figure('Lambda1')
    # plt.imshow(Lambda1)
    # plt.figure('Lambda2')
    # plt.imshow(Lambda2)
    # plt.show()

    # 保存图像
    mpimg.imsave(PROCESS_ROOT + '\Lambda1_sigma=%.1f.png' % sigma, Lambda1)
    mpimg.imsave(PROCESS_ROOT + '\Lambda2_sigma=%.1f.png' % sigma, Lambda2)
    return Lambda1, Lambda2


def gradient2(F, option):
    '''返回option维度的梯度
    F: 图像
    option: 可以为'x', 'y'
    '''
    [k, l] = F.shape
    D = np.zeros(F.shape)
    if (option == 'x'):
        D[1:k - 2, :] = (F[2:k - 1, :] - F[0:k - 3, :]) / 2  # 中心差商代替导数
        D[0, :] = D[1, :]
        D[k - 1, :] = D[k - 2, :]
        pass
    elif (option == 'y'):
        D[:, 1:l - 2] = (F[:, 2:l - 1] - F[:, 0:l - 3]) / 2
        D[:, 0] = D[:, 1]
        D[:, l - 1] = D[:, l - 2]
        pass
    else:
        print('Unknown option')
    return D


def Hessian2D(I, Sigma=1, spacing=[1, 1]):
    '''返回图像I的海森矩阵中的Dxx, Dyy, Dxy

    调用：imgaussian'''
    if (Sigma > 0):
        F = imgaussian(I, Sigma, spacing)  # 高斯滤波
        # # debug
        # plt.figure('gaussian')
        # plt.imshow(F)
        # plt.show()
        # 保存图像
        mpimg.imsave(PROCESS_ROOT + 'Gaussian_sigma=%.1f.png' % Sigma, F)
    else:
        F = I

    Dy = gradient2(F, 'y')  # 计算梯度
    Dyy = gradient2(Dy, 'y')
    Dx = gradient2(F, 'x')
    Dxx = gradient2(Dx, 'x')
    Dxy = gradient2(Dx, 'y')
    # # debug
    # plt.figure('Dxx')
    # plt.imshow(Dxx)
    # plt.figure('Dxy')
    # plt.imshow(Dxy)
    # plt.figure('Dyy')
    # plt.imshow(Dyy)
    # plt.show()
    return Dxx, Dyy, Dxy


def imgaussian(I, Sigma, spacing, siz=None):
    '''对图像高斯滤波
    siz: 核的大小，必须为正奇数。

    调用：cv2.GaussionBlur
    这里没有考虑spacing，默认为[1,1]。如果考虑spacing，需要使用cv2.filter2D'''
    if (siz == None):
        siz = np.ceil(Sigma * 6)  # 核的大小（不知道为什么是这样，应该只是为了保证核够大吧）
        if (siz % 2 == 0):
            siz = siz + 1  # 这一步是为了让siz为奇数。matlab版本里没有判断，好像偶数也可以，但是这里貌似不行。

    if (Sigma > 0):
        # I = GaussianBlur(I, [siz, siz], Sigma) # new style getargs format but argument is not a tuple
        siz = int(siz)  # GaussianBlur函数必须输入整数siz
        I = GaussianBlur(I, (siz, siz), Sigma)

    return I


def eigvalOfHessian2D(Dxx, Dxy, Dyy):
    '''通过Dxx,Dxy,Dyy计算海森矩阵特征值'''

    # 特征方程的√Δ
    tmp = np.sqrt(np.power(Dxx - Dyy, 2) + 4 * np.power(Dxy, 2))

    # 特征值(用的是(-b±√Δ))/2a
    mu1 = (Dxx + Dyy + tmp) / 2
    mu2 = (Dxx + Dyy - tmp) / 2

    # 排序abs(Lambda1)<abs(Lambda2)
    # mu1和mu2都是向量，所以不是这么排的……
    # if(abs(mu1)>abs(mu2)):
    #     Lambda1 = mu1
    #     Lambda2 = mu2
    # else:
    #     Lambda2 = mu1
    #     Lambda1 = mu2
    # return Lambda1, Lambda2

    Lambda1 = np.copy(mu1)  # 要用深复制，不然下面λ1和λ2的值会一样！
    Lambda2 = np.copy(mu2)
    check = np.abs(mu1) > np.abs(mu2)
    Lambda1[check] = mu2[check]
    Lambda2[check] = mu1[check]
    # 【最令人迷惑的是，其实所有的mu1都大于mu2，这张图实际上不需要排序
    # 但我还没想明白为什么。除非所有Dxx + Dyy >= 0，但总觉得没道理啊】
    # # debug
    # plt.figure('mu1')
    # plt.imshow(mu1)
    # plt.figure('mu2')
    # plt.imshow(mu2)
    # plt.show()
    return Lambda1, Lambda2


