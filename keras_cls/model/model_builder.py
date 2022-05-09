from keras_cls.model.efficientnet import EfficientNet
from keras_cls.model.resnet import ResNet
from keras_cls.utils.regularization import add_regularization


def get_model(args, num_class):
    if args.backbone[0:3] == "Res":
        try:
            depth = int(args.backbone[-3:])
        except:
            depth = int(args.backbone[-2:])
        model = ResNet(classes=num_class,
                       type=depth,
                       concat_max_and_average_pool=args.concat_max_and_average_pool,
                       weights=args.weights,
                       loss=args.loss)
    elif args.backbone[0:3] == "Eff":
        model = EfficientNet(classes=num_class,
                             type=args.backbone[-2:],
                             concat_max_and_average_pool=args.concat_max_and_average_pool,
                             weights=args.weights,
                             loss=args.loss,
                             dropout=args.dropout)
    else:
        raise ValueError("{} is not supported!".format(args.backbone))
    model = model.get_model()
    # if  args.optimizer != "AdamW":
    #     model = add_regularization(model,args.weight_decay)

    return model
