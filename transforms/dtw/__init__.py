from transforms.dtw.dtw_layer import DTWLayer, DTWLayerPerChannel, DTWFeatures, DTWFeatures_c

dtw_mode = {"dtw": DTWLayer, "dtw_c": DTWLayerPerChannel, "dtwfeats": DTWFeatures, "dtwfeats_c": DTWFeatures_c}
