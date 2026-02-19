import os, yaml, sys
import numpy as np
from scipy.spatial.distance import pdist, squareform
import pytest
ENV = os.getenv("MY_ENV", "dev")
with open("../../config.yaml", "r") as f:
    config = yaml.safe_load(f)
paths = config[ENV]["paths"]
sys.path.append(paths["src_path"])
from general_utils.utils import print_wise, get_lagplot, autocorr_mat, create_RDM, spearman, compute_dRSA, nan_check, choose_summary_stat, get_lagplot_subset, index_gram, cosine_sim, subsample_RDM

# --- Tests for get_lagplot_checks (indirect because we don't export it) ---

def test_lagplot_checks_index_error():
    corr_mat = np.zeros((5, 5))
    with pytest.raises(IndexError):
        get_lagplot(corr_mat, max_lag=10, min_datapts=2, symmetric=False)

def test_lagplot_checks_warning_for_min_datapts(capfd): # captures file descriptors
    corr_mat = np.ones((8, 8))
    get_lagplot(corr_mat, max_lag=6, min_datapts=5, symmetric=False)
    captured = capfd.readouterr()
    assert "datapoints used to compute extreme offsets" in captured.out

def test_lagplot_checks_warning_for_nans(capfd):
    corr_mat = np.eye(5)
    corr_mat[1, 1] = np.nan
    get_lagplot(corr_mat, max_lag=2, min_datapts=2, symmetric=False)
    captured = capfd.readouterr()
    assert "There are nans" in captured.out


# --- Tests for autocorr_mat ---

def test_autocorr_mat_auto():
    data = np.random.randn(4, 10)
    corr_mat = autocorr_mat(data)
    assert corr_mat.shape == (10, 10)
    assert np.allclose(np.diag(corr_mat), 1, atol=1e-8)

def test_autocorr_mat_cross():
    data1 = np.random.randn(4, 10)
    data2 = np.random.randn(4, 10)
    corr_mat = autocorr_mat(data1, data2)
    assert corr_mat.shape == (10, 10)


# --- Tests for get_lagplot ---

def test_get_lagplot_symmetric():
    data = np.random.randn(4, 20)
    corr_mat = autocorr_mat(data)
    lagplot = get_lagplot(corr_mat, max_lag=5, min_datapts=3, symmetric=True)
    assert lagplot.shape == (6,)   # max_lag + 1
    assert not np.isnan(lagplot).all()
    assert lagplot[0] == 1

def test_get_lagplot_asymmetric():
    data1 = np.random.randn(4, 20)
    data2 = np.random.randn(4, 20)
    corr_mat = autocorr_mat(data1, data2)
    lagplot = get_lagplot(corr_mat, max_lag=5, min_datapts=3, symmetric=False)
    assert lagplot.shape == (11,)  # 2*max_lag + 1
    assert not np.isnan(lagplot).all()



# -----------------------------------------------------------
# TESTS FOR create_RDM
# -----------------------------------------------------------

def test_create_rdm_correlation_matches_pdist():
    """Check that correlation RDM matches scipy's pdist 'correlation'."""
    X = np.random.randn(5, 10)  # (features × datapoints)
    
    rdm_custom = create_RDM(X, metric='correlation')
    rdm_scipy  = pdist(X.T, metric='correlation')
    
    assert np.allclose(rdm_custom, rdm_scipy, atol=1e-12)


def test_create_rdm_output_length():
    """Check that the vectorized RDM has correct length."""
    X = np.random.randn(4, 6)
    rdm = create_RDM(X)
    
    n = X.shape[1]
    expected_len = n * (n - 1) // 2  # upper triangular
    assert rdm.shape[0] == expected_len


def test_create_rdm_custom_metric():
    """Check that passing a callable metric works."""
    def my_metric(a, b):
        return np.sum(np.abs(a - b))  # L1
    
    X = np.random.randn(3, 8)
    rdm = create_RDM(X, metric=my_metric)
    
    rdm_expected = pdist(X.T, metric=my_metric)
    assert np.allclose(rdm, rdm_expected)

# -----------------------------------------------------------
# TESTS FOR spearman
# -----------------------------------------------------------

def test_spearman_perfect_correlation():
    """Should be +1 for perfectly increasing monotonic relation."""
    x = np.array([1, 2, 3, 4])
    y = np.array([10, 20, 30, 40])
    rho = spearman(x, y)
    assert np.isclose(rho, 1.0)


def test_spearman_perfect_negative_correlation():
    """Should be -1 for perfectly decreasing monotonic relation."""
    x = np.array([1, 2, 3])
    y = np.array([30, 20, 10])
    rho = spearman(x, y)
    assert np.isclose(rho, -1.0)


def test_spearman_against_scipy():
    """Compare against scipy.stats.spearmanr."""
    from scipy.stats import spearmanr

    x = np.random.randn(20)
    y = np.random.randn(20)

    rho_custom = spearman(x, y)
    rho_scipy, _ = spearmanr(x, y)

    assert np.isclose(rho_custom, rho_scipy, atol=1e-12)

# -----------------------------------------------------------
# TESTS FOR compute_dRSA
# -----------------------------------------------------------

def test_compute_dRSA_shape():
    """Check that output shape is (time, time)."""
    neural = np.random.randn(4, 5, 10)  # channels, time, trials
    model  = np.random.randn(4, 5, 10)
    
    dRSA = compute_dRSA(neural, model)
    
    assert dRSA.shape == (5, 5)


def test_compute_dRSA_metric_RDM_model_none_uses_same_metric():
    """Ensure metric_RDM_model=None makes neural + model use same metric."""
    neural = np.random.randn(3, 4, 6)
    model  = np.random.randn(3, 4, 6)

    dRSA1 = compute_dRSA(neural, model, metric_RDM='correlation', metric_RDM_model=None)
    dRSA2 = compute_dRSA(neural, model, metric_RDM='correlation', metric_RDM_model='correlation')

    assert np.allclose(dRSA1, dRSA2)


def test_compute_dRSA_with_custom_metric():
    """Check that custom metric for dRSA works."""
    def my_metric(a, b):
        return np.dot(a, b)  # simple similarity

    neural = np.random.randn(4, 5, 6)
    model  = np.random.randn(4, 5, 6)

    dRSA = compute_dRSA(neural, model, metric_dRSA=my_metric)

    assert dRSA.shape == (5, 5)


from io import StringIO

# --- Tests for nan_check ---

def test_nan_check_no_nan(capsys):
    """nan_check should print nothing when there are no NaN values."""
    corr = np.array([[1, 2], [3, 4]])
    nan_check(corr)
    captured = capsys.readouterr()
    assert captured.out == ""


def test_nan_check_with_nan(capsys):
    """nan_check should print a warning when NaN values are present."""
    corr = np.array([[1, np.nan], [3, 4]])
    nan_check(corr)
    captured = capsys.readouterr()
    assert "There are nans in corr_mat" in captured.out
    

# --- Tests for choose_summary_stat ---

def test_choose_summary_stat_mean():
    """Should return np.nanmean when summary_stat='mean'."""
    stat = choose_summary_stat('mean')
    assert stat is np.nanmean


def test_choose_summary_stat_median():
    """Should return np.nanmedian when summary_stat='median'."""
    stat = choose_summary_stat('median')
    assert stat is np.nanmedian


def test_choose_summary_stat_invalid():
    """Should raise ValueError for invalid summary_stat."""
    with pytest.raises(ValueError):
        choose_summary_stat('std')


def test_basic_identity_matrix():
    """Diagonal lags of identity matrix should be 1 at tau=0 and 0 elsewhere."""
    corr = np.eye(5)
    max_lag = 2

    out = get_lagplot_subset(corr, neural_idx=range(5), max_lag=max_lag)

    # length check
    assert len(out) == 2 * max_lag + 1

    center = max_lag
    # tau = 0
    assert out[center] == 1.0
    # tau = ±1, ±2 -> no overlapping ones
    assert np.all(out[np.arange(len(out)) != center] == 0.0)


def test_mean_vs_median():
    """Check switching the statistic works."""
    corr = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ])
    neural_idx = [0, 1, 2]

    out_mean = get_lagplot_subset(corr, neural_idx, max_lag=1, summary_stat="mean")
    out_median = get_lagplot_subset(corr, neural_idx, max_lag=1, summary_stat="median")

    # center (tau=0): mean and median of diag [1,5,9]
    assert out_mean[1] == pytest.approx(np.mean([1, 5, 9]))
    assert out_median[1] == pytest.approx(np.median([1, 5, 9]))


def test_custom_indices():
    """Check the function respects neural_idx and model_idx."""
    corr = np.arange(25).reshape(5, 5)

    neural_idx = [1, 2, 3]
    model_idx  = [0, 1, 2, 3, 4]

    out = get_lagplot_subset(corr, neural_idx, model_idx, max_lag=1)

    # tau = 0 → diag values are corr[i,i]
    expected_tau0 = [corr[i, i] for i in neural_idx]
    assert out[1] == np.mean(expected_tau0)

    # tau = -1 → corr[i, i+1]
    expected_tau1 = [corr[i, i+1] for i in neural_idx]
    assert out[0] == np.mean(expected_tau1)


def test_out_of_bounds_raises_error():
    """If no valid elements exist for a given lag, an IndexError should be raised."""
    corr = np.eye(3)

    with pytest.raises(IndexError):
        get_lagplot_subset(corr, neural_idx=[0, 1, 2], max_lag=5)


def test_nan_handling():
    """NaNs should be ignored thanks to np.nanmean / np.nanmedian."""
    corr = np.array([
        [1, np.nan],
        [3, 4]
    ])
    with pytest.warns(RuntimeWarning, match="Mean of empty slice"): # because np.nanmean(np.nan)
        out = get_lagplot_subset(corr, neural_idx=[0, 1], max_lag=1)

    # tau = 0 → diag [1, 4] ⇒ mean = 2.5
    assert out[1] == pytest.approx(2.5)


def test_min_datapts_warning(monkeypatch):
    """If < min_datapts, print_wise should be called."""
    corr = np.eye(5)

    # Mock print_wise to track calls
    called = {"flag": False}
    def fake_print(msg):
        called["flag"] = True

    monkeypatch.setattr("general_utils.utils.print_wise", fake_print)

    get_lagplot_subset(corr, neural_idx=[1, 2], max_lag=2, min_datapts=5)

    assert called["flag"] is True


def test_compare_lagplot_lagplot_subset():
    """ checks that get_lagplot_subset falls back to get_lagplot once we consider all datapoints """
    a = np.random.random((30,30))
    b = get_lagplot_subset(a,range(a.shape[0]), max_lag=10, min_datapts=1)#, range(0,30))
    c = get_lagplot(a, max_lag=10, min_datapts=1)
    assert np.all(np.equal(b,c)) # all vals should be equal 


def test_zero_prediction():
    a = np.random.random((30,30))
    rows, cols = np.triu_indices(a.shape[0], k=1)  # k=0 includes diagonal, k=1 excludes
    a[rows, cols] = 0 # Substitute values, e.g., set upper triangle to 0, i.e. all the entries in which t_neu<t_mod (prediction)
    b = get_lagplot_subset(a,range(a.shape[0])[10:15], max_lag=10, min_datapts=1)#, range(0,30))
    c = get_lagplot(a, max_lag=10, min_datapts=1)
    assert np.all(np.equal(b[:10], np.zeros(10)))
    assert np.all(np.equal(c[:10], np.zeros(10)))


@pytest.mark.parametrize("n", [2, 3, 5, 10])
def test_index_gram_length(n):
    gram = np.random.randn(n, n)
    out = index_gram(gram)
    assert out.shape == (n * (n - 1) // 2,)

def test_index_gram_matches_numpy_triu():
    n = 6
    gram = np.random.randn(n, n)

    rows, cols = np.triu_indices(n, k=1)
    ref = gram[rows, cols]

    out = index_gram(gram)

    assert np.allclose(out, ref)


def test_index_gram_symmetric_matrix():
    n = 5
    A = np.random.randn(n, n)
    gram = (A + A.T) / 2

    out = index_gram(gram)
    ref = squareform(gram, checks=False)

    assert np.allclose(out, ref)


def test_index_gram_ignores_diagonal():
    gram = np.eye(4)
    out = index_gram(gram)
    assert np.all(out == 0)


@pytest.mark.parametrize("n,d", [(5, 3), (10, 2), (20, 8)])
def test_cosine_sim_shape(n, d):
    x = np.random.randn(d, n)
    out = cosine_sim(x)
    assert out.shape == (n * (n - 1) // 2,)


def test_cosine_sim_matches_scipy():
    d, n = 6, 10
    x = np.random.randn(d, n)

    ref = pdist(x.T, metric="cosine")
    out = cosine_sim(x)

    assert np.allclose(out, ref, atol=1e-6)


def test_cosine_sim_self_zero():
    x = np.random.randn(5, 7)
    gram = 1 - (x / np.linalg.norm(x, axis=0)).T @ (x / np.linalg.norm(x, axis=0))

    vec = index_gram(gram)
    assert np.all(vec >= 0)


def test_cosine_sim_identical_vectors():
    v = np.random.randn(5, 1)
    x = np.repeat(v, 4, axis=1)

    out = cosine_sim(x)
    assert np.allclose(out, 0)


def test_cosine_sim_orthogonal():
    x = np.eye(4)
    out = cosine_sim(x)
    assert np.allclose(out, 1)


def test_cosine_sim_scale_invariance():
    x = np.random.randn(5, 6)
    out1 = cosine_sim(x)
    out2 = cosine_sim(10 * x)
    assert np.allclose(out1, out2)


def test_cosine_sim_zero_vector():
    x = np.random.randn(5, 4)
    x[:, 0] = 0

    out = cosine_sim(x)
    assert np.any(np.isnan(out))

def test_subsample_rdm_shape():
    RDM = np.random.randn(10, 10)
    idx = [1, 3, 5, 7]

    out = subsample_RDM(RDM, idx)

    assert out.shape == (len(idx), len(idx))

def test_subsample_rdm_values():
    RDM = np.arange(100).reshape(10, 10)
    idx = [2, 4, 9]

    out = subsample_RDM(RDM, idx)

    ref = RDM[np.ix_(idx, idx)]
    assert np.array_equal(out, ref)

def test_subsample_rdm_order():
    RDM = np.arange(25).reshape(5, 5)
    idx = [4, 1, 3]

    out = subsample_RDM(RDM, idx)

    ref = RDM[np.ix_(idx, idx)]
    assert np.array_equal(out, ref)

def test_subsample_rdm_duplicate_indices():
    RDM = np.random.randn(6, 6)
    idx = [1, 1, 4]

    out = subsample_RDM(RDM, idx)

    assert out.shape == (3, 3)
    assert np.allclose(out[0], out[1])

def test_subsample_rdm_single_index():
    RDM = np.random.randn(5, 5)
    idx = [3]

    out = subsample_RDM(RDM, idx)

    assert out.shape == (1, 1)
    assert out[0, 0] == RDM[3, 3]

def test_subsample_rdm_full_indices():
    n = 7
    RDM = np.random.randn(n, n)
    idx = list(range(n))

    out = subsample_RDM(RDM, idx)

    assert np.array_equal(out, RDM)

def test_subsample_rdm_empty_indices():
    RDM = np.random.randn(5, 5)
    idx = []

    out = subsample_RDM(RDM, idx)

    assert out.shape == (0, 0)


def test_subsample_rdm_symmetry():
    A = np.random.randn(6, 6)
    RDM = (A + A.T) / 2
    idx = [0, 2, 5]

    out = subsample_RDM(RDM, idx)

    assert np.allclose(out, out.T)

def test_subsample_rdm_invalid_indices():
    RDM = np.random.randn(5, 5)
    idx = [0, 6]

    with pytest.raises(IndexError):
        subsample_RDM(RDM, idx)
