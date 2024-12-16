#%%
def precision_score(tp, fp):
    """Calculate precision."""
    divider = tp + fp
    return tp / divider if divider > 0 else 0

#%%
def precision_corrected_score(tp, fp, tnrt):
    """Calculate precision."""
    divider = tp + fp
    return tp * tnrt / divider if divider > 0 else 0

#%%
def recall_score(tp, fn):
    """Calculate recall."""
    divider = tp + fn
    return tp / divider if divider > 0 else 0

#%%
def f05_score(precision, recall):
    """Calculate F0.5 score."""
    beta = 0.5
    divider = beta**2 * precision + recall
    return (1 + beta**2) * (precision * recall) / divider if divider > 0 else 0


#%%
def ranges_overlap(start1, end1, start2, end2):
    return not (end1 < start2 or end2 < start1)

#%%
def process_pred_channels(raw_pred_channels, influence_limit = 0.0):
        pred_channels = []
        for el in raw_pred_channels:
            _aux = el.split(" ")
            channel, influnce = _aux[0], float(_aux[1].strip("()%"))
            if influnce > influence_limit:
                pred_channels.append(channel)
        return pred_channels