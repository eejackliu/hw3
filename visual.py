import torch
import torchvision
import torchvision.transforms as T
import random
import numpy as np
from scipy.ndimage.filters import gaussian_filter1d
import matplotlib.pyplot as plt
from cs231n.image_utils import SQUEEZENET_MEAN, SQUEEZENET_STD
from PIL import Image

plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'
def preprocess(img, size=224):
    transform = T.Compose([
        T.Resize(size),
        T.ToTensor(),
        T.Normalize(mean=SQUEEZENET_MEAN.tolist(),
                    std=SQUEEZENET_STD.tolist()),
        T.Lambda(lambda x: x[None]),
    ])
    return transform(img)

def deprocess(img, should_rescale=True):
    transform = T.Compose([
        T.Lambda(lambda x: x[0]),
        T.Normalize(mean=[0, 0, 0], std=(1.0 / SQUEEZENET_STD).tolist()),
        T.Normalize(mean=(-SQUEEZENET_MEAN).tolist(), std=[1, 1, 1]),
        T.Lambda(rescale) if should_rescale else T.Lambda(lambda x: x),
        T.ToPILImage(),
    ])
    return transform(img)
def rescale(x):
    low, high = x.min(), x.max()
    x_rescaled = (x - low) / (high - low)
    return x_rescaled

def blur_image(X, sigma=1):
    X_np = X.cpu().clone().numpy()
    X_np = gaussian_filter1d(X_np, sigma, axis=2)
    X_np = gaussian_filter1d(X_np, sigma, axis=3)
    X.copy_(torch.Tensor(X_np).type_as(X))
    return X
model = torchvision.models.squeezenet1_1(pretrained=True)
import torch.nn.functional as F
# We don't want to train the model, so tell PyTorch not to compute gradients
# with respect to model parameters.
for param in model.parameters():
    param.requires_grad = False
from cs231n.data_utils import load_imagenet_val
X, y, class_names = load_imagenet_val(num=5)

# plt.figure(figsize=(12, 6))
# for i in range(5):
#     plt.subplot(1, 5, i + 1)
#     plt.imshow(X[i])
#     plt.title(class_names[y[i]])
#     plt.axis('off')
# plt.gcf().tight_layout()
# Example of using gather to select one entry from each row in PyTorch
# def gather_example():
#     N, C = 4, 5
#     s = torch.randn(N, C)
#     y = torch.LongTensor([1, 2, 1, 3])
#     print(s)
#     print(y)
#     print(s.gather(1, y.view(-1, 1)).squeeze())
# gather_example()

def compute_saliency_maps(X, y, model):
    """
    Compute a class saliency map using the model for images X and labels y.

    Input:
    - X: Input images; Tensor of shape (N, 3, H, W)
    - y: Labels for X; LongTensor of shape (N,)
    - model: A pretrained CNN that will be used to compute the saliency map.

    Returns:
    - saliency: A Tensor of shape (N, H, W) giving the saliency maps for the input
    images.
    """

    # Make sure the model is in "test" mode
    model.eval()
     # Make input tensor require gradient
    X.requires_grad_()
    saliency = None
    ##############################################################################
    # TODO: Implement this function. Perform a forward and backward pass through #
    # the model to compute the gradient of the correct class score with respect  #
    # to each input image. You first want to compute the loss over the correct   #
    # scores (we'll combine losses across a batch by summing), and then compute  #
    # the gradients with a backward pass.                                        #
    ##############################################################################
    out=model(X)
    loss=F.cross_entropy(out,y)
    loss.backward()
    saliency=torch.abs(X.grad)
    argmax=torch.argmax(saliency,dim=1,keepdim=True)
    saliency=saliency.gather(1,argmax).squeeze()

    ##############################################################################
    #                             END OF YOUR CODE                               #
    ##############################################################################
    return saliency
def show_saliency_maps(X, y):
    # Convert X and y from numpy arrays to Torch Tensors
    X_tensor = torch.cat([preprocess(Image.fromarray(x)) for x in X], dim=0)
    y_tensor = torch.LongTensor(y)

    # Compute saliency maps for images in X
    saliency = compute_saliency_maps(X_tensor, y_tensor, model)

    # Convert the saliency map from Torch Tensor to numpy array and show images
    # and saliency maps together.
    saliency = saliency.numpy()
    print(saliency)
    N = X.shape[0]
    for i in range(N):
        plt.subplot(2, N, i + 1)
        plt.imshow(X[i])
        plt.axis('off')
        plt.title(class_names[y[i]])
        plt.subplot(2, N, N + i + 1)
        plt.imshow(saliency[i], cmap=plt.cm.hot)
        plt.axis('off')
        plt.gcf().set_size_inches(12, 5)
    plt.show()
# show_saliency_maps(X, y)
def gradient_ascent_saliency_maps(X, y):
    # Convert X and y from numpy arrays to Torch Tensors
    X_tensor = torch.cat([preprocess(Image.fromarray(x)) for x in X], dim=0)
    y_tensor = torch.LongTensor(y)
    for i in range(200):
    # Compute saliency maps for images in X
        saliency = compute_saliency_maps(X_tensor, y_tensor, model)
        X_tensor=X_tensor.detach()+saliency.unsqueeze(1)
    # Convert the saliency map from Torch Tensor to numpy array and show images
    # and saliency maps together.
    saliency = saliency.numpy()
    X_tensor=[np.asarray(deprocess(x.unsqueeze(0),should_rescale=False)) for x in X_tensor]
    # xx=[x for x in X_tensor]
    # print (xx[1])
    # X_tensor=deprocess(X_tensor,should_rescale=False)
    N = X.shape[0]
    for i in range(N):
        plt.subplot(2, N, i + 1)
        plt.imshow(X[i])
        plt.axis('off')
        plt.title(class_names[y[i]])
        plt.subplot(2, N, N + i + 1)
        # plt.imshow(saliency[i], cmap=plt.cm.hot)
        plt.imshow(X_tensor[i])
        plt.axis('off')
        plt.gcf().set_size_inches(12, 5)
    plt.show()
# gradient_ascent_saliency_maps(X,y)
#not true the negative gradient will be positive because abs(),which would not amplify the picture
def make_fooling_image(X, target_y, model):
    """
    Generate a fooling image that is close to X, but that the model classifies
    as target_y.

    Inputs:
    - X: Input image; Tensor of shape (1, 3, 224, 224)
    - target_y: An integer in the range [0, 1000)
    - model: A pretrained CNN

    Returns:
    - X_fooling: An image that is close to X, but that is classifed as target_y
    by the model.
    """
    # Initialize our fooling image to the input image, and make it require gradient
    X_fooling = X.clone()
    X_fooling = X_fooling.requires_grad_()

    learning_rate = 1
    ##############################################################################
    # TODO: Generate a fooling image X_fooling that the model will classify as   #
    # the class target_y. You should perform gradient ascent on the score of the #
    # target class, stopping when the model is fooled.                           #
    # When computing an update step, first normalize the gradient:               #
    #   dX = learning_rate * g / ||g||_2                                         #
    #                                                                            #
    # You should write a training loop.                                          #
    #                                                                            #
    # HINT: For most examples, you should be able to generate a fooling image    #
    # in fewer than 100 iterations of gradient ascent.                           #
    # You can print your progress over iterations to check your algorithm.       #
    ##############################################################################
    model.eval()
    for i in range(50):
        out=model(X_fooling)
        loss=F.cross_entropy(out,torch.LongTensor([target_y]))
        # loss=out[0][target_y]
        loss.backward()
        # we can use both loss=out[0][target_y] and cross_entroy ,the different is that the former one is that we want the value to be maximum ,the other one we want the value to become minimum
        # so the first we need to ascent the gradient ,the last we need to descent the gradient .Hence we need to maximum the fooling class score and minimum the crose-entropy between the input and the fooling class
        # with torch.no_grad():
        #     dx=learning_rate*X_fooling.grad/(X_fooling.norm()**2)
        #     X_fooling+=dx
        g=X_fooling.grad
        tmp=learning_rate*g/g.norm()**2
        X_fooling=X_fooling-tmp
        X_fooling=X_fooling.detach()
        X_fooling.requires_grad_()
        print(loss)
        if loss<1:
            break
    ##############################################################################
    #                             END OF YOUR CODE                               #
    ##############################################################################
    return X_fooling
idx = 0
target_y = 6

X_tensor = torch.cat([preprocess(Image.fromarray(x)) for x in X], dim=0)
X_fooling = make_fooling_image(X_tensor[idx:idx+1], target_y, model)

scores = model(X_fooling)
assert target_y == scores.data.max(1)[1][0].item(), 'The model is not fooled!'
X_fooling_np = deprocess(X_fooling.clone())
X_fooling_np = np.asarray(X_fooling_np).astype(np.uint8)

plt.subplot(1, 4, 1)
plt.imshow(X[idx])
plt.title(class_names[y[idx]])
plt.axis('off')

plt.subplot(1, 4, 2)
plt.imshow(X_fooling_np)
plt.title(class_names[target_y])
plt.axis('off')

plt.subplot(1, 4, 3)
X_pre = preprocess(Image.fromarray(X[idx]))
diff = np.asarray(deprocess(X_fooling - X_pre, should_rescale=False))
plt.imshow(diff)
plt.title('Difference')
plt.axis('off')

plt.subplot(1, 4, 4)
diff = np.asarray(deprocess(10 * (X_fooling - X_pre), should_rescale=False))
plt.imshow(diff)
plt.title('Magnified difference (10x)')
plt.axis('off')

plt.gcf().set_size_inches(12, 5)
plt.show()
