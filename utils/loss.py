import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class FocalLoss(nn.Module):
    r"""
        This criterion is a implemenation of Focal Loss, which is proposed in
        Focal Loss for Dense Object Detection.

            Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])

        The losses are averaged across observations for each minibatch.

        Args:
            alpha(1D Tensor, Variable) : the scalar factor for this criterion
            gamma(float, double) : gamma > 0; reduces the relative loss for well-classiﬁed examples (p > .5),
                                   putting more focus on hard, misclassiﬁed examples
            size_average(bool): By default, the losses are averaged over observations for each minibatch.
                                However, if the field size_average is set to False, the losses are
                                instead summed for each minibatch.


    """
    def __init__(self, gamma=2, size_average=True):
        super(FocalLoss, self).__init__()
        # if alpha is None:
        #     self.alpha = Variable(torch.ones(class_num, 1))
        # else:
        #     if isinstance(alpha, Variable):
        #         self.alpha = alpha
        #     else:
        #         self.alpha = Variable(alpha)
        self.gamma = gamma
        self.size_average = size_average

    def forward(self, inputs, targets):
        N = inputs.size(0)
        C = inputs.size(1)
        P = F.softmax(inputs)

        class_mask = inputs.data.new(N, C).fill_(0)
        class_mask = Variable(class_mask)
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.)
        #print(class_mask)


        # if inputs.is_cuda and not self.alpha.is_cuda:
        #     self.alpha = self.alpha.cuda()
        #alpha = self.alpha[ids.data.view(-1)]

        probs = (P*class_mask).sum(1).view(-1,1)

        log_p = probs.log()
        #print('probs size= {}'.format(probs.size()))
        #print(probs)

        batch_loss = -torch.pow((1-probs), self.gamma)*log_p
        #print('-----bacth_loss------')
        #print(batch_loss)


        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss

class p_Loss (nn.Module,):

    def __init__(self,size_avg):
        super (p_Loss, self).__init__ ()
        self.size_avg = size_avg
        self.gamma =1

    def forward(self, probs):
        probs = torch.clamp (probs, 1e-10, 1)
        log_p = probs.log()
        #batch_loss = -log_p
        batch_loss = -torch.pow ((1 - probs), self.gamma) * log_p
        if self.size_avg:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss

class TripletLoss(nn.Module):
    def __init__(self,inputs, targets):
        super(TripletLoss, self).__init__()
        self.alpha = 4

    def forward(self, inputs, targets):
        n = inputs.size(0)
        # Compute pairwise distance
        targets = targets.cuda()
        dist_mat = self.alpha*euclidean_dist(inputs)
        # split the positive and negative pairs
        eyes_ = Variable(torch.eye(n, n)).cuda()
        # eyes_ = Variable(torch.eye(n, n))
        pos_mask = targets.expand(n, n).eq(targets.expand(n, n).t())

        neg_mask = eyes_.eq(eyes_) - pos_mask


        pos_mat = pos_mask.float() * dist_mat
        neg_mat = neg_mask.float() * dist_mat


        neg_mask = neg_mask + eyes_.eq(1)


        pos_mat = pos_mat.masked_fill(neg_mask,10)
        neg_mat = neg_mat.masked_fill(pos_mask,10)


        pos_pair = torch.sort(pos_mat,1)[0]
        pos_mask_2 = pos_pair<10
        neg_pair = torch.sort(neg_mat,1)[0]
        triplet_mat = torch.log(torch.exp(pos_pair - neg_pair) + 1)*pos_mask_2.float()

        loss = torch.sum(triplet_mat)

        return loss

def euclidean_dist(inputs_):
    n = inputs_.size(0)
    dist = torch.pow(inputs_, 2).sum(dim=1, keepdim=True).expand(n, n)
    dist = dist + dist.t()
    dist.addmm_(1, -2, inputs_, inputs_.t())
    # for numerical stability
    dist = dist.clamp(min=1e-12).sqrt()
    return dist

def log_sum_exp(x):
    # See implementation detail in
    # http://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/
    # b is a shift factor. see link.
    # x.size() = [N, C]:
    b, _ = torch.max(x, 1)
    y = b + torch.log(torch.exp(x - b.expand_as(x)).sum(1))
    # y.size() = [N, 1]. Squeeze to [N] and return
    return y.squeeze(1)


def class_select(logits, target):
    # in numpy, this would be logits[:, target].
    batch_size, num_classes = logits.size()
    if target.is_cuda:
        device = target.data.get_device()
        one_hot_mask = torch.autograd.Variable(torch.arange(0, num_classes)
                                               .long()
                                               .repeat(batch_size, 1)
                                               .cuda(device)
                                               .eq(target.data.repeat(num_classes, 1).t()))
    else:
        one_hot_mask = torch.autograd.Variable(torch.arange(0, num_classes)
                                               .long()
                                               .repeat(batch_size, 1)
                                               .eq(target.data.repeat(num_classes, 1).t()))
    return logits.masked_select(one_hot_mask)


def cross_entropy_with_weights(logits, target, weights=None):
    assert logits.dim() == 2
    assert not target.requires_grad
    target = target.squeeze(1) if target.dim() == 2 else target
    assert target.dim() == 1
    loss = log_sum_exp(logits) - class_select(logits, target)
    if weights is not None:
        # loss.size() = [N]. Assert weights has the same shape
        assert list(loss.size()) == list(weights.size())
        # Weight the loss
        loss = loss * weights
    return loss


class CrossEntropyLoss_with_weights(nn.Module):
    """
    Cross entropy with instance-wise weights. Leave `aggregate` to None to obtain a loss
    vector of shape (batch_size,).
    """
    def __init__(self, aggregate='mean'):
        super(CrossEntropyLoss_with_weights, self).__init__()
        assert aggregate in ['sum', 'mean', None]
        self.aggregate = aggregate

    def forward(self, input, target, weights=None):
        if self.aggregate == 'sum':
            return cross_entropy_with_weights(input, target, weights).sum()
        elif self.aggregate == 'mean':
            return cross_entropy_with_weights(input, target, weights).mean()
        elif self.aggregate is None:
            return cross_entropy_with_weights(input, target, weights)


"""SVM loss
Weston and Watkins version multiclass hinge loss @ https://en.wikipedia.org/wiki/Hinge_loss
for each sample, given output (a vector of n_class values) and label y (an int \in [0,n_class-1])
loss = sum_i(max(0, (margin - output[y] + output[i]))^p) where i=0 to n_class-1 and i!=y
Note: hinge loss is not differentiable
      Let's denote hinge loss as h(x)=max(0,1-x). h'(x) does not exist when x=1, 
      because the left and right limits do not converge to the same number, i.e.,
      h'(1-delta)=-1 but h'(1+delta)=0.

      To overcome this obstacle, people proposed squared hinge loss h2(x)=max(0,1-x)^2. In this case,
      h2'(1-delta)=h2'(1+delta)=0
"""

class multiClassHingeLoss(nn.Module):
    def __init__(self, p=1, margin=1, weight=None, size_average=True):
        super(multiClassHingeLoss, self).__init__()
        self.p=p
        self.margin=margin
        self.weight=weight#weight for each class, size=n_class, variable containing FloatTensor,cuda,reqiures_grad=False
        self.size_average=size_average
    def forward(self, output, y):#output: batchsize*n_class
        #print(output.requires_grad)
        #print(y.requires_grad)
        output_y=output[torch.arange(0,y.size()[0]).long().cuda(),y.data.cuda()].view(-1,1)#view for transpose
        #margin - output[y] + output[i]
        loss=output-output_y+self.margin#contains i=y
        #remove i=y items

        loss[torch.arange(0,y.size()[0]).long().cuda(),y.data.cuda()]=0
        #max(0,_)
        loss[loss<0]=0
        #^p
        if(self.p!=1):
            loss=torch.pow(loss,self.p)
        #add weight
        if(self.weight is not None):
            loss=loss*self.weight
        #sum up
        loss=torch.sum(loss)
        if(self.size_average):
            loss/=output.size()[0]#output.size()[0]
        return loss