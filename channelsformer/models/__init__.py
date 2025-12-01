from channelsformer.models.channelsformer import ChannelSFormer


def build_cmv_model(config, **kwargs):
    model_type = config.MODEL.TYPE
    if model_type in ["channelsformer"]:
        model = ChannelSFormer(
            img_size=config.DATA.IMG_SIZE,
            patch_size=config.MODEL.CHANNELSFORMER.PATCH_SIZE,
            in_chans=config.MODEL.CHANNELSFORMER.IN_CHANS,
            num_classes=config.MODEL.NUM_CLASSES,
            embed_dim=config.MODEL.CHANNELSFORMER.EMBED_DIM,
            depth=config.MODEL.CHANNELSFORMER.DEPTHS,
            num_heads=config.MODEL.CHANNELSFORMER.NUM_HEADS,
            attention_type=config.MODEL.CHANNELSFORMER.ATTENTION_TYPE,
            drop_rate=config.MODEL.DROP_RATE,
            drop_path_rate=config.MODEL.DROP_PATH_RATE,
            use_channel_embedding=config.MODEL.CHANNELSFORMER.USE_CHANNEL_EMBEDDING,
            separate_cls_for_channel=config.MODEL.CHANNELSFORMER.SEPARATE_CLS_FOR_CHANNEL,
            separate_cls_aggregation=config.MODEL.CHANNELSFORMER.SEPARATE_CLS_AGGREGATION,
            SPACE_CHANNEL_ORDER=config.MODEL.CHANNELSFORMER.SPACE_CHANNEL_ORDER,
            no_additional_mapping=config.MODEL.CHANNELSFORMER.NO_ADDITIONAL_MAPPING,
            separate_cls_init=config.MODEL.CHANNELSFORMER.SEPARATE_CLS_INIT,
            use_channel_embedding_for_cls=config.MODEL.CHANNELSFORMER.use_channel_embedding_FOR_CLS,
        )
        return model

    return None


def build_model(config, is_pretrain=False):
    model = None
    if model is None:
        model = build_cmv_model(config)
    return model
