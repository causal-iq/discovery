
def test_bruno_metrics():

    def metrics(tot, acc, sens, spec):

        print(('\n\n{} records with accuracy {:.5f}, sensitivity {:.5f}, and ' +
               'specificity {:.5f}').format(tot, acc, sens, spec))

        bad = (1.0 - acc) * tot
        ok = acc * tot

        c1 = (1 - spec) * ok / spec
        c2 = (1 - spec) / spec

        _tp = round(sens * (bad - c1) / (1 - sens - c2 * sens))
        _tn = round((acc * tot) - _tp)
        _fp = round(_tn * (1.0 - spec) / spec)
        _fn = round(_tp * (1.0 - sens) / sens)

        print('\nComputed TP = {}, FP = {}, FN = {}, TN = {}'
              .format(_tp, _fp, _fn, _tn))

        prec = _tp / (_tp + _fp)
        rec = _tp / (_tp + _fn)
        f1 = 2 * prec * rec / (prec + rec)
        print('Positive precision is {:.3f}, recall {:.3f} and F1 {:.3f}'
              .format(prec, rec, f1))

        prec = _tn / (_tn + _fn)
        rec = _tn / (_tn + _fp)
        f1 = 2 * prec * rec / (prec + rec)
        print('Negative precision is {:.3f}, recall {:.3f} and F1 {:.3f}'
              .format(prec, rec, f1))

    tp = 190
    fp = 100
    fn = 10
    tn = 700

    print('\n\nTP = {}, FP = {}, FN = {}, TN = {}'.format(tp, fp, fn, tn))

    tot = tp + fp + tn + fn
    acc = (tp + tn) / tot
    sens = tp / (tp + fn)
    spec = tn / (tn + fp)

    metrics(tot, acc, sens, spec)

    metrics(1000000, 0.6915, 0.7677, 0.6886)
