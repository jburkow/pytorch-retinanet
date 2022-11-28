import math

import torch
import torch.nn as nn
from retinanet import losses
from retinanet.anchors import Anchors
from retinanet.model import PyramidFeatures
from retinanet.utils import BasicBlock, BBoxTransform, Bottleneck, ClipBoxes
from torchvision.ops import nms


class FiLMGenerator(nn.Module):
    """
    MLP that generates FiLM parameters (gains and biases).
    
    Attributes
    ----------
    n_features : int
        Number of non-image feature inputs.
    n_channels : int
        Number of feature maps to modulate (also, half the number of MLP outputs: n_channels gains + n_channels biases).
    n_hidden_features : int
        Number of units in the single hidden layer. By default, set to n_channels // 2.
    """

    def __init__(self, n_features, n_channels=256, n_hidden_features=None):
        super(FiLMGenerator, self).__init__()
        self.n_features = n_features
        self.n_channels = n_channels
        self.n_hidden_features = n_hidden_features if n_hidden_features is None else self.n_channels // 2

        # Simple MLP to predict gains and biases from non-image inputs
        # Potential improvements: Dropout after each linear layer, LeakyReLU instead of ReLU
        self.film_generator = nn.Sequential(
            nn.Linear(self.n_features, self.n_hidden_features),
            # nn.Dropout(0.1),
            nn.ReLU(inplace=True),
            nn.Linear(self.n_hidden_features, self.n_hidden_features),
            # nn.Dropout(0.2),
            nn.ReLU(inplace=True),
            nn.Linear(self.n_hidden_features, 2*self.n_channels)
        )

    def forward(self, x):
        # Input shape: (batch_size, n_features)
        # Output shape: (batch_size, 2*n_channels)... later decomposed to (batch_size, n_channels) gammas and (batch_size, n_channels) betas
        film_params = self.film_generator(x)

        return film_params


class FiLMLayer(nn.Module):
    """Layer that performs Featurewise Linear Modulation (FiLM)."""
    def __init__(self):
        super(FiLMLayer, self).__init__()

    def forward(self, F, gammas, betas):
        # Repeat (tile) gammas and betas to match shape of feature maps in F: from shape (batch_size, n_channels) -> (batch_size, n_channels, height, width)
        gammas = torch.stack([gammas]*F.shape[2], dim=2)
        gammas = torch.stack([gammas]*F.shape[3], dim=3)
        betas = torch.stack([betas]*F.shape[2], dim=2)
        betas = torch.stack([betas]*F.shape[3], dim=3)

        return (1 + gammas) * F + betas


class FiLMedRegressionModel(nn.Module):
    def __init__(self, num_features_in, num_anchors=9, feature_size=256):
        super(FiLMedRegressionModel, self).__init__()

        self.film = FiLMLayer()

        self.conv1 = nn.Conv2d(num_features_in, feature_size, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()

        self.conv2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()

        self.conv3 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act3 = nn.ReLU()

        self.conv4 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act4 = nn.ReLU()

        self.output = nn.Conv2d(feature_size, num_anchors * 4, kernel_size=3, padding=1)

    def forward(self, x, gammas, betas):
        # There are infinite ways to do this, but this version applies FiLM once after the first convolution in the regression head
        out = self.conv1(x)
        out = self.film(out, gammas, betas)  # APPLY FiLM!
        out = self.act1(out)

        out = self.conv2(out)
        out = self.act2(out)

        out = self.conv3(out)
        out = self.act3(out)

        out = self.conv4(out)
        out = self.act4(out)

        out = self.output(out)

        # out is B x C x W x H, with C = 4*num_anchors
        out = out.permute(0, 2, 3, 1)

        return out.contiguous().view(out.shape[0], -1, 4)


class FiLMedClassificationModel(nn.Module):
    def __init__(self, num_features_in, num_anchors=9, num_classes=80, prior=0.01, feature_size=256, FiLMed=False):
        super(FiLMedClassificationModel, self).__init__()

        self.num_classes = num_classes
        self.num_anchors = num_anchors

        self.film = FiLMLayer()

        self.conv1 = nn.Conv2d(num_features_in, feature_size, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()

        self.conv2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()

        self.conv3 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act3 = nn.ReLU()

        self.conv4 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act4 = nn.ReLU()

        self.output = nn.Conv2d(feature_size, num_anchors * num_classes, kernel_size=3, padding=1)
        self.output_act = nn.Sigmoid()

    def forward(self, x, gammas, betas):
        # There are infinite ways to do this, but this version applies FiLM once after the first convolution in the classification head
        out = self.conv1(x)
        out = self.film(out, gammas, betas)  # APPLY FiLM!
        out = self.act1(out)

        out = self.conv2(out)
        out = self.act2(out)

        out = self.conv3(out)
        out = self.act3(out)

        out = self.conv4(out)
        out = self.act4(out)

        out = self.output(out)
        out = self.output_act(out)

        # out is B x C x W x H, with C = n_classes + n_anchors
        out1 = out.permute(0, 2, 3, 1)

        batch_size, width, height, channels = out1.shape

        out2 = out1.view(batch_size, width, height, self.num_anchors, self.num_classes)

        return out2.contiguous().view(x.shape[0], -1, self.num_classes)


class FiLMedResNet(nn.Module):
    """FiLMed version of RetinaNet. FiLM is applied after the first convolution block in the regression and classification head, once for each of the 5 feature pyramid outputs (i.e., 5*2=10 FiLM layers)."""

    def __init__(self, num_classes, block, layers):
        super(FiLMedResNet, self).__init__()
        # Initialize FiLM generators: 10 in total for this specific configuration.
        # One FiLM generator (MLP) for each of the 5 feature pyramid outputs (P2-P7) that will be fed through the classification head
        self.cls_film_generators = nn.ModuleList([FiLMGenerator(n_features=5, n_hidden_features=128, n_channels=256) for _ in range(5)])
        # One FiLM generator (MLP) for each of the 5 feature pyramid outputs (P2-P7) that will be fed through the regression head
        self.reg_film_generators = nn.ModuleList([FiLMGenerator(n_features=5, n_hidden_features=128, n_channels=256) for _ in range(5)])

        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        if block == BasicBlock:
            fpn_sizes = [self.layer2[layers[1] - 1].conv2.out_channels, self.layer3[layers[2] - 1].conv2.out_channels,
                         self.layer4[layers[3] - 1].conv2.out_channels]
        elif block == Bottleneck:
            fpn_sizes = [self.layer2[layers[1] - 1].conv3.out_channels, self.layer3[layers[2] - 1].conv3.out_channels,
                         self.layer4[layers[3] - 1].conv3.out_channels]
        else:
            raise ValueError(f"Block type {block} not understood")

        self.fpn = PyramidFeatures(fpn_sizes[0], fpn_sizes[1], fpn_sizes[2])

        # Initialize FiLMed classification and regression heads!
        self.regressionModel = FiLMedRegressionModel(256, feature_size=256)
        self.classificationModel = FiLMedClassificationModel(256, feature_size=256, num_classes=num_classes)

        self.anchors = Anchors()

        self.regressBoxes = BBoxTransform()

        self.clipBoxes = ClipBoxes()

        self.focalLoss = losses.FocalLoss()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        prior = 0.01

        self.classificationModel.output.weight.data.fill_(0)
        self.classificationModel.output.bias.data.fill_(-math.log((1.0 - prior) / prior))

        self.regressionModel.output.weight.data.fill_(0)
        self.regressionModel.output.bias.data.fill_(0)

        self.freeze_bn()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        layers.extend(block(self.inplanes, planes) for _ in range(1, blocks))
        return nn.Sequential(*layers)

    def freeze_bn(self):
        '''Freeze BatchNorm layers.'''
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()

    def forward(self, inputs):
        if self.training:
            img_batch, metadata_batch, annotations = inputs
        else:
            img_batch, metadata_batch = inputs

        # Generate FiLM parameters for classification head
        cls_betas = []
        cls_gammas = []
        for film_generator in self.cls_film_generators:
            film_params = film_generator(metadata_batch)

            # Split output into two "chunks": (batch_size, n_channels) gammas and (batch_size, n_channels) betas
            betas, gammas = torch.split(film_params, film_generator.n_channels, dim=1)
            
            cls_betas.append(betas)
            cls_gammas.append(gammas)
        # Create (5, batch_size, n_channels) tensor of gammas and betas, respectively, for classification head (5 for each of the feature pyramid outputs)
        cls_betas = torch.stack(cls_betas)
        cls_gammas = torch.stack(cls_gammas)

        # Generate FiLM parameters for regression head
        reg_betas = []
        reg_gammas = []
        for film_generator in self.reg_film_generators:
            film_params = film_generator(metadata_batch)

            # Split output into two "chunks": (batch_size, n_channels) gammas and (batch_size, n_channels) betas
            betas, gammas = torch.split(film_params, film_generator.n_channels, dim=1)

            reg_betas.append(betas)
            reg_gammas.append(gammas)
        # Create (5, batch_size, n_channels) tensor of gammas and betas, respectively, for regression head (5 for each of the feature pyramid outputs)
        reg_betas = torch.stack(reg_betas)
        reg_gammas = torch.stack(reg_gammas)

        x = self.conv1(img_batch)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        features = self.fpn([x2, x3, x4])

        # Feed feature pyramid output + associated FiLM params through regression head
        regression = torch.cat([self.regressionModel(features[i], reg_gammas[i], reg_betas[i]) for i in range(len(features))], dim=1)

        # Feed feature pyramid output + associated FiLM params through classification head
        classification = torch.cat([self.classificationModel(features[i], cls_gammas[i], cls_betas[i]) for i in range(len(features))], dim=1)

        anchors = self.anchors(img_batch)

        if self.training:
            return self.focalLoss(classification, regression, anchors, annotations)
        else:
            transformed_anchors = self.regressBoxes(anchors, regression)
            transformed_anchors = self.clipBoxes(transformed_anchors, img_batch)

            finalResult = [[], [], []]

            finalScores = torch.Tensor([])
            finalAnchorBoxesIndexes = torch.Tensor([]).long()
            finalAnchorBoxesCoordinates = torch.Tensor([])

            if torch.cuda.is_available():
                finalScores = finalScores.cuda()
                finalAnchorBoxesIndexes = finalAnchorBoxesIndexes.cuda()
                finalAnchorBoxesCoordinates = finalAnchorBoxesCoordinates.cuda()

            for i in range(classification.shape[2]):
                scores = torch.squeeze(classification[:, :, i])
                scores_over_thresh = (scores > 0.05)
                if scores_over_thresh.sum() == 0:
                    # return empty tensors to count toward true negatives
                    return torch.Tensor(), torch.Tensor(), torch.Tensor()
                    # no boxes to NMS, just continue
                    # continue

                scores = scores[scores_over_thresh]
                anchorBoxes = torch.squeeze(transformed_anchors)
                anchorBoxes = anchorBoxes[scores_over_thresh]
                anchors_nms_idx = nms(anchorBoxes, scores, 0.5)

                finalResult[0].extend(scores[anchors_nms_idx])
                finalResult[1].extend(torch.tensor([i] * anchors_nms_idx.shape[0]))
                finalResult[2].extend(anchorBoxes[anchors_nms_idx])

                finalScores = torch.cat((finalScores, scores[anchors_nms_idx]))
                finalAnchorBoxesIndexesValue = torch.tensor([i] * anchors_nms_idx.shape[0])
                if torch.cuda.is_available():
                    finalAnchorBoxesIndexesValue = finalAnchorBoxesIndexesValue.cuda()

                finalAnchorBoxesIndexes = torch.cat((finalAnchorBoxesIndexes, finalAnchorBoxesIndexesValue))
                finalAnchorBoxesCoordinates = torch.cat((finalAnchorBoxesCoordinates, anchorBoxes[anchors_nms_idx]))

            return [finalScores, finalAnchorBoxesIndexes, finalAnchorBoxesCoordinates]
