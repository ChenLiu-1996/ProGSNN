from .attribute_hashmap import AttributeHashmap


def parse_hparams(hparams: AttributeHashmap) -> AttributeHashmap:
    gpu_ids = []
    for item in hparams.gpu_ids.split(','):
        if len(item) > 0:
            gpu_ids.append(int(item))
    hparams.gpu_ids = tuple(gpu_ids)
    return hparams
