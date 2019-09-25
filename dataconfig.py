# -*- coding: utf-8 -*-

# Preprocess data
preprocessing = {
    'missing_data': False,
    # 'missing_data': 'none',
    'header': False,
    'classification_index': 1,
    'features_start_index': 2,
    'features_end_index': -1,
    'batch_size': 0.6,
    'shuffle_seed': 1000000,

    # Skipping ID, Class, (7,17,27) is the f) feature that is a mix of two other features
    'to_skip': [],
}
