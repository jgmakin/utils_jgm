# standard libraries
from functools import wraps
import string
import pdb
try:
    from collections.abc import Sequence
except ImportError:
    from collections import Sequence
from inspect import Signature, Parameter, getfullargspec
import re
from IPython import display

# third-party packages
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
from scipy.stats.mstats import zscore
try:
    import samplerate
except:  ### ModuleNotFoundError:
    print("Warning: package 'samplerate' not found; skipping")


# see "The Tau Manifesto"
tau = 2*np.pi

# A_cubehelix
A_cubehelix = np.array([
    [-0.14861,  1.78277, 1.0],
    [-0.29227, -0.90649, 1.0],
    [ 1.97294,  0.0, 1.0]
])


'''
:Author: J.G. Makin (except where otherwise noted)
'''


# Cribbed from Aaron Hall and Ashwini Chaudhary, here:
#   https://codereview.stackexchange.com/questions/173045/
class MutableNamedTuple(Sequence):
    """Abstract Base Class for objects as efficient as mutable
    namedtuples.
    Subclass and define your named fields with __slots__.
    """

    @classmethod
    def get_signature(cls):
        parameters = [
            Parameter(name=slot, kind=Parameter.POSITIONAL_OR_KEYWORD) for slot in cls.__slots__
        ]
        return Signature(parameters=parameters)

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        slots = cls.__slots__
        cls.__slots__ = tuple(slots.split()) if isinstance(slots, str) else tuple(slots)
        cls.__signature__ = cls.get_signature()
        cls.__init__.__signature__ = cls.get_signature()
        cls.__doc__ = '{cls.__name__}{cls.__signature__}\n\n{cls.__doc__}'.format(
            cls=cls)

    def __new__(cls, *args, **kwargs):
        if cls is MutableNamedTuple:
            raise TypeError("Can't instantiate abstract class MutableNamedTuple")
        return super().__new__(cls)

    @classmethod
    def _get_bound_args(cls, args, kwargs):
        return Signature.bind(cls.__signature__, *args, **kwargs).arguments.items()

    __slots__ = ()

    def __init__(self, *args, **kwargs):
        bound_args = self._get_bound_args(args, kwargs)
        for slot, value in bound_args:
            setattr(self, slot, value)

    def __repr__(self):
        return type(self).__name__ + repr(tuple(self))

    def __iter__(self):
        for name in self.__slots__:
            yield getattr(self, name)

    def __getitem__(self, index):
        return getattr(self, self.__slots__[index])

    def __len__(self):
        return len(self.__slots__)


# See https://stackoverflow.com/questions/3888158/
def auto_attribute(function=None, *, CHECK_MANIFEST=False):

    def _auto_attribute(fxn):
        @wraps(fxn)
        def wrapped(self, *args, **kwargs):
            _assign_args(self, list(args), kwargs, fxn, CHECK_MANIFEST)
            fxn(self, *args, **kwargs)
        return wrapped

    if function:
        return _auto_attribute(function)
    else:
        return _auto_attribute


# Cribbed from Enteleform here: https://stackoverflow.com/questions/1389180
# Modified slightly to accommodate loading from a manifest.
def _assign_args(
    instance, args, kwargs, function, CHECK_MANIFEST=False
):
    (POSITIONAL_PARAMS, VARIABLE_PARAM, _, KEYWORD_DEFAULTS, _,
     KEYWORD_ONLY_DEFAULTS, _) = getfullargspec(function)
    POSITIONAL_PARAMS = POSITIONAL_PARAMS[1:]  # remove 'self'
    if CHECK_MANIFEST:
        manifest_index = POSITIONAL_PARAMS.index('manifest')
        manifest = args.pop(manifest_index)
        POSITIONAL_PARAMS.pop(manifest_index)
    else:
        manifest = dict()

    def set_attribute(instance, parameter, default_arg):
        if not(parameter.startswith("_")):
            if default_arg is None and parameter in manifest.keys():
                setattr(instance, parameter, manifest[parameter])
            else:
                setattr(instance, parameter, default_arg)

    def assign_keyword_defaults(parameters, defaults):
        for parameter, default_arg in zip(
            reversed(parameters), reversed(defaults)
        ):
            set_attribute(instance, parameter, default_arg)

    def assign_positional_args(parameters, args):
        for parameter, arg in zip(parameters, args.copy()):
            set_attribute(instance, parameter, arg)
            args.remove(arg)

    def assign_keyword_args(kwargs):
        for parameter, arg in kwargs.items():
            set_attribute(instance, parameter, arg)

    def assign_keyword_only_defaults(defaults):
        return assign_keyword_args(defaults)

    def assign_variable_args(parameter, args):
        set_attribute(instance, parameter, args)

    if KEYWORD_DEFAULTS:
        assign_keyword_defaults(
            parameters=POSITIONAL_PARAMS, defaults=KEYWORD_DEFAULTS)
    if KEYWORD_ONLY_DEFAULTS:
        assign_keyword_only_defaults(defaults=KEYWORD_ONLY_DEFAULTS)
    if args:
        assign_positional_args(parameters=POSITIONAL_PARAMS, args=args)
    if kwargs:
        assign_keyword_args(kwargs=kwargs)
    if VARIABLE_PARAM:
        assign_variable_args(parameter=VARIABLE_PARAM, args=args)


# https://stackoverflow.com/questions/5967500/
def natural_keys(text):
    return [int(c) if c.isdigit() else c for c in re.split(r'(\d+)', text)]


# https://stackoverflow.com/questions/11517986/
def barplot_annotate_brackets(
    num1, num2, data, center, extrema, yerr=None, dh=.05, barh=.05, fs=None,
    max_asterisk=None, ABOVE=True
):
    """
    Annotate barplot with p-values.

    :param num1: number of left bar to put bracket over
    :param num2: number of right bar to put bracket over
    :param data: string to write or number for generating asterixes
    :param center: x coords of all bars/boxes (like plt.bar() input)
    :param extrema: heights or depths of all bars/boxes (like plt.bar() input)
    :param yerr: yerrs of all bars (like plt.bar() input)
    :param dh: height offset over bar / bar + yerr in axes coordinates (0 to 1)
    :param barh: bar height in axes coordinates (0 to 1)
    :param fs: font size
    :param max_asterisk: maximum number of asterisks to write (for very small
        p-values)
    :param ABOVE: whether to place bracket above or below the bars/boxes.  NB
        that when using this option, the extrema should be set to *depths*.
    """

    if type(data) is str:
        text = data
    else:
        # * is p < 0.05
        # ** is p < 0.005
        # *** is p < 0.0005
        # etc.
        text = ''
        p = .05

        while data < p:
            text += '*'
            p /= 10.

            if max_asterisk and len(text) == max_asterisk:
                break

        if len(text) == 0:
            text = 'n. s.'

    lx, ly = center[num1], extrema[num1]
    rx, ry = center[num2], extrema[num2]

    ax_y0, ax_y1 = plt.gca().get_ylim()
    dh *= (ax_y1 - ax_y0)
    barh *= (ax_y1 - ax_y0)

    if ABOVE:
        # put the bracket on top
        if yerr:
            ly += yerr[num1]
            ry += yerr[num2]
        y = max(ly, ry) + dh
        y_bracket = y + barh
    else:
        # put the bracket underneath
        if yerr:
            ly -= yerr[num1]
            ry -= yerr[num2]
        y = min(ly, ry) - dh
        y_bracket = y - barh

    barx = [lx, lx, rx, rx]
    bary = [y, y_bracket, y_bracket, y]
    mid = ((lx+rx)/2, y_bracket)

    plt.plot(barx, bary, c='black')

    kwargs = dict(ha='center', va='bottom')
    if fs is not None:
        kwargs['fontsize'] = fs

    plt.text(*mid, text, **kwargs)


def cosine_similarity(X, Y=None):
    '''
    :X: data matrix, (Ndims x Nexamples)
    :Y: data matrix, (Ndims x Mexamples)

    Returns the cosine similarity between all pairs of examples in data matrix
    X; or, if the optional data matrix Y is also passed, between the elements
    of X and Y, but not within either matrix.

    NB that the data matrix is in "linear-systems" format (each column is an
    example), not "psychology" format (each row is an example)!
    '''

    if Y is None:
        norms = np.sqrt(np.sum(X**2, axis=0, keepdims=True))
        return X.T@X/(norms.T@norms)
    else:
        normsX = np.sqrt(np.sum(X**2, axis=0, keepdims=True))
        normsY = np.sqrt(np.sum(Y**2, axis=0, keepdims=True))
        return X.T@Y/(normsX.T@normsY)


def rescale(X, xmin, xmax, zmin, zmax):
    '''
    Transforms x in [xmin, xmax] into z in [zmin, zmax] using an
     affine rescaling.

    NB: X must have size Nexamples x Ndims, where Ndims is the common
     length of xmin, xmax, zmin, and zmax.
    '''

    # vectorize
    xmin = np.reshape(xmin, [1, -1])
    zmin = np.reshape(zmin, [1, -1])
    xmax = np.reshape(xmax, [1, -1])
    zmax = np.reshape(zmax, [1, -1])

    # lots of implicit expansion
    column_scaling = (zmax - zmin)/(xmax - xmin)

    return column_scaling*(X - xmin) + zmin


def prime_factorize(n):
    # prime factorization by trial division, copied verbatim from:
    #   https://stackoverflow.com/questions/16996217/
    factor_list = []
    d = 2
    while d*d <= n:
        while (n % d) == 0:
            factor_list.append(d)
            n //= d
        d += 1
    if n > 1:
        factor_list.append(n)
    return factor_list


def close_factors(n, num_factors):
    factor_list = prime_factorize(n)

    if num_factors < 1:
        raise ValueError("User must request at least one factor!")

    if len(factor_list) <= num_factors:
        factor_list += [1]*(num_factors - len(factor_list))
        factor_list.sort()
        return factor_list
    else:
        while len(factor_list) > num_factors:
            a = factor_list.pop(0)
            b = factor_list.pop(0)
            factor_list = [a*b] + factor_list
            factor_list.sort()
        return factor_list


# ...
def draw_confusion_matrix(matrix, axis_labels, figsize):
    '''Draw confusion matrix for MNIST.'''
    import tfmpl
    
    fig = tfmpl.create_figure(figsize=figsize)
    ax = fig.add_subplot(111)
    ax.set_title('Confusion matrix')

    tfmpl.plots.confusion_matrix.draw(
        ax, matrix,
        axis_labels=axis_labels,
        normalize=True
    )
    return fig


# ...
def heatmap_confusions(
        fig, p_of_x_given_y, x_axis_labels=None, y_axis_labels=None):

    # get the axis labels
    x_range, y_range = p_of_x_given_y.shape
    ax = fig.add_subplot(1, 1, 1)
    x_axis_labels = x_axis_labels if x_axis_labels else np.arange(x_range)
    y_axis_labels = y_axis_labels if y_axis_labels else np.arange(y_range)

    # select out the subset for which there are non-zero y's
    y_is_non_zero = np.sum(p_of_x_given_y, axis=1) > 0
    x_axis_labels = np.array(x_axis_labels)[y_is_non_zero]
    y_axis_labels = np.array(y_axis_labels)[y_is_non_zero]
    p_of_x_given_y = p_of_x_given_y[np.ix_(y_is_non_zero, y_is_non_zero)]

    # clear the axes and plot with colorbar
    cax = ax.imshow(p_of_x_given_y)
    ax.grid(False)
    your_colorbar = fig.colorbar(cax)

    # labels
    plt.xlabel('q')
    plt.ylabel('p')
    plt.xticks(np.arange(x_range), x_axis_labels, rotation=90)
    plt.yticks(np.arange(y_range), y_axis_labels)

    fig.tight_layout()

    return fig


def fig_to_rgb_array(fig, EXPAND=True):
    # from stackoverflow.com/questions/38543850, answered by
    # yauheni_selivonchyk
    fig.canvas.draw()
    buf = fig.canvas.tostring_rgb()
    ncols, nrows = fig.canvas.get_width_height()
    shape = (nrows, ncols, 3) if not EXPAND else (1, nrows, ncols, 3)
    return np.frombuffer(buf, dtype=np.uint8).reshape(shape)


def rgb_to_cmyk(RGB_matrix, cmyk_scale=100):
    CMY_matrix = 1 - RGB_matrix/255.
    k = np.min(CMY_matrix, axis=1, keepdims=True)
    CMY_matrix = (CMY_matrix-k)/(1-k)
    return cmyk_scale*np.concatenate((CMY_matrix, k), axis=1)


def params2cubehelix(start, fraction, saturation, rotation):

    # derived parameters
    M = saturation*fraction*(1.0 - fraction)/2.0
    phi = tau*(start/3.0 + rotation*fraction + 1.0)
    v = np.array([
        [M*np.cos(phi)],
        [M*np.sin(phi)],
        [fraction]
    ])

    # return
    return A_cubehelix@v


def cubehelix2params(RGB_color, rotation):

    if type(RGB_color) in [list, tuple]:
        RGB_color = np.array(RGB_color)[:, None]

    # invert the transformation
    y = np.linalg.inv(A_cubehelix)@RGB_color

    # cartesian to polar
    phi = np.mod(np.arctan2(y[1, 0], y[0, 0]), tau)
    M = (y[0, 0]**2 + y[1, 0]**2)**(1/2)

    # derived to underived params
    fraction = y[2, 0]
    saturation = 2*M/(fraction*(1-fraction))
    start = 3.0*(phi/tau - rotation*fraction - 1)

    return start, fraction, saturation


def wer_vector(
    references, hypotheses, m_cost=0, s_cost=1, i_cost=1, d_cost=1, cost_fxn=None
):
    """
    Vectorized calculation of word error rate with Levenshtein distance.

    Works only for iterables up to 254 elements (uint8).
    O(nm) time and space complexity.

    Input arguments:
    --------
        references : list of lists of words
        hypotheses : list of lists of words

    Returns:
    --------
        numpy array (vector) of len(references) (== len(hypotheses))

    Example
    --------
    >>> wer(["well that's great news".split(),
             "who is there".split(),
             "who is there".split(),
             "".split(),
            ],
            "that's the fine news we wanted".split(),
            "is there".split(),
            "".split(),
            "who is there".split(),
            ])
    outputs: [5 1 3 3]

    This vectorized version is indeed faster than its scalar counterpart for as
    few as about 20 sentences.

    Created: 02/12/18
        by JGM
        Inspired by scalar version found here:
            https://martin-thoma.com/word-error-rate-calculation/
    """

    # ...
    N_sentences = len(references)
    if len(hypotheses) != len(references):
        raise ValueError('no. of references must equal no. of hypotheses')

    reference_lengths = [len(reference) for reference in references]
    hypothesis_lengths = [len(hypothesis) for hypothesis in hypotheses]
    N_ref_max = max(reference_lengths)
    N_hyp_max = max(hypothesis_lengths)
    d_max = max(N_ref_max, N_hyp_max)

    # you have to put the references and hypotheses into a matrix
    #  of size Ncases x N_*_max.
    reference_matrix = np.zeros((N_sentences, N_ref_max), dtype='object')
    hypothesis_matrix = np.zeros((N_sentences, N_hyp_max), dtype='object')
    for i_sentence, (reference, hypothesis) in enumerate(
            zip(references, hypotheses)):
        reference_matrix[i_sentence, 0:len(reference)] = reference
        hypothesis_matrix[i_sentence, 0:len(hypothesis)] = hypothesis

    # initialize
    if cost_fxn is None:
        def cost_fxn(ref, hyp):
            return m_cost, s_cost, i_cost, d_cost

        distance_tensor = np.zeros((N_sentences, N_ref_max + 1, N_hyp_max + 1),
                                   dtype=np.uint8)
        distance_tensor[:, 0] = np.indices((N_hyp_max + 1, ))
        distance_tensor[:, :, 0] = np.indices((N_ref_max + 1, ))
    else:
        distance_tensor = np.zeros((N_sentences, N_ref_max + 1, N_hyp_max + 1))
        distance_tensor += np.inf
        distance_tensor[:, 0, 0] = 0

    # compute minimum edit distance
    for i_ref in range(0, N_ref_max):
        for i_hyp in range(0, N_hyp_max):
            m_cost, s_cost, i_cost, d_cost = cost_fxn(
                reference_matrix[:, i_ref], hypothesis_matrix[:, i_hyp])
            match = m_cost + distance_tensor[:, i_ref, i_hyp] + d_max*(
                reference_matrix[:, i_ref] != hypothesis_matrix[:, i_hyp])
            substitution = s_cost + distance_tensor[:, i_ref, i_hyp]
            insertion = i_cost + distance_tensor[:, i_ref + 1, i_hyp]
            deletion = d_cost + distance_tensor[:, i_ref, i_hyp + 1]
            distance_tensor[:, i_ref+1, i_hyp+1] = np.minimum.reduce(
                [match, substitution, insertion, deletion])

    distances = distance_tensor[(np.arange(N_sentences),
                                 reference_lengths, hypothesis_lengths)]
    return distances/reference_lengths


def wer(references, hypotheses):
    """
    Calculation of WER with Levenshtein distance.

    Works only for iterables up to 254 elements (uint8).
    O(nm) time ans space complexity.

    Parameters
    ----------
    reference : list
    hypothesis : list

    Returns
    -------
    int

    Examples
    --------
    >>> wer("who is there".split(), "is there".split())
    1
    >>> wer("who is there".split(), "".split())
    3
    >>> wer("".split(), "who is there".split())
    3
    """

    #-------------------------------------------------------------------------#
    # Revised: 02/12/18
    #   -rationalized
    # Cribbed: 02/12/18
    #   from https://martin-thoma.com/word-error-rate-calculation/
    #-------------------------------------------------------------------------#

    distances = []

    for reference, hypothesis in zip(references, hypotheses):
        N_ref = len(reference)
        N_hyp = len(hypothesis)
        d_max = max(N_ref, N_hyp)

        # initialisation
        distance = np.zeros((N_ref+1, N_hyp+1), dtype=np.uint8)
        distance[0, :] = np.arange(N_hyp+1)
        distance[:, 0] = np.arange(N_ref+1)
        # pdb.set_trace()

        # computation
        for i_ref in range(0, N_ref):
            for i_hyp in range(0, N_hyp):
                if reference[i_ref] == hypothesis[i_hyp]:
                    distance[i_ref+1, i_hyp+1] = distance[i_ref][i_hyp]
                else:
                    substitution = distance[i_ref, i_hyp] + 1
                    insertion = distance[i_ref+1, i_hyp] + 1
                    deletion = distance[i_ref, i_hyp+1] + 1
                    distance[i_ref+1, i_hyp+1] = min(
                        substitution, insertion, deletion)

        distances.append(distance[N_ref, N_hyp])

    # return distance[N_ref][N_hyp]
    return distances


def wer_two(ref, hyp, debug=False):
    #  wer_two      Another word-error-rate calculator
    #-------------------------------------------------------------------------#
    # Cribbed: 02/12/18
    #   from
    # http://progfruits.blogspot.com/2014/02/word-error-rate-wer-and-word.html
    #-------------------------------------------------------------------------#
    SUB_PENALTY = 1
    DEL_PENALTY = 1
    INS_PENALTY = 1

    r = ref.split()
    h = hyp.split()
    # costs will holds the costs, like in the Levenshtein distance algorithm
    costs = [[0 for inner in range(len(h) + 1)] for outer in range(len(r) + 1)]
    # backtrace will hold the operations we've done.
    # so we could later backtrace, like the WER algorithm requires us to.
    backtrace = [[0 for inner in range(len(h) + 1)]
                 for outer in range(len(r) + 1)]

    OP_OK = 0
    OP_SUB = 1
    OP_INS = 2
    OP_DEL = 3

    # First column represents the case where we achieve zero
    # hypothesis words by deleting all reference words.
    for i in range(1, len(r) + 1):
        costs[i][0] = DEL_PENALTY * i
        backtrace[i][0] = OP_DEL

    # First row represents the case where we achieve the hypothesis
    # by inserting all hypothesis words into a zero-length reference.
    for j in range(1, len(h) + 1):
        costs[0][j] = INS_PENALTY * j
        backtrace[0][j] = OP_INS

    # computation
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            if r[i - 1] == h[j - 1]:
                costs[i][j] = costs[i - 1][j - 1]
                backtrace[i][j] = OP_OK
            else:
                # penalty is always 1
                substitutionCost = costs[i - 1][j - 1] + SUB_PENALTY
                insertionCost = costs[i][j - 1] + INS_PENALTY
                deletionCost = costs[i - 1][j] + DEL_PENALTY

                costs[i][j] = min(substitutionCost,
                                  insertionCost, deletionCost)
                if costs[i][j] == substitutionCost:
                    backtrace[i][j] = OP_SUB
                elif costs[i][j] == insertionCost:
                    backtrace[i][j] = OP_INS
                else:
                    backtrace[i][j] = OP_DEL

    # back trace though the best route:
    i = len(r)
    j = len(h)
    numSub = 0
    numDel = 0
    numIns = 0
    numCor = 0
    if debug:
        print("OP\tREF\tHYP")
        lines = []
    while i > 0 or j > 0:
        if backtrace[i][j] == OP_OK:
            numCor += 1
            i -= 1
            j -= 1
            if debug:
                lines.append("OK\t" + r[i] + "\t" + h[j])
        elif backtrace[i][j] == OP_SUB:
            numSub += 1
            i -= 1
            j -= 1
            if debug:
                lines.append("SUB\t" + r[i] + "\t" + h[j])
        elif backtrace[i][j] == OP_INS:
            numIns += 1
            j -= 1
            if debug:
                lines.append("INS\t" + "****" + "\t" + h[j])
        elif backtrace[i][j] == OP_DEL:
            numDel += 1
            i -= 1
            if debug:
                lines.append("DEL\t" + r[i] + "\t" + "****")
    if debug:
        lines = reversed(lines)
        for line in lines:
            print(line)
        print("#cor " + str(numCor))
        print("#sub " + str(numSub))
        print("#del " + str(numDel))
        print("#ins " + str(numIns))
    return (numSub + numDel + numIns) / (float)(len(r))
    wer_result = round((numSub + numDel + numIns) / (float)(len(r)), 3)
    return {'WER': wer_result, 'Cor': numCor, 'Sub': numSub,
            'Ins': numIns, 'Del': numDel}


def pseudomode(num_list):
    '''
    Suppose a list of numbers tends to be dominated by just one entry--but
    sometimes by two.  Suppose further than these numbers are almost always
    neighbors (e.g., consecutive integers).  The sensible representative number
    is the mode, but when two entries are precisely equally frequent, one might
    prefer to get the mean of these entries, rather than just picking one (or
    worse, as in the case of statistics.mode, throwing an exception!).
    '''
    count_dict = {num: list(num_list).count(num) for num in set(num_list)}
    max_count = max(count_dict.values())
    return np.mean(
        [key for (key, value) in count_dict.items() if value == max_count])


def information_transfer_rate(p_correct, num_classes, delta_t=1.0):
    '''
    A common *approximation* to the actual information transfer rate, i.e. the
    mutual information between a random categorical variable and the output
    of a classifier, divided by the amount of time required for classification.
    See e.g.:
    https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0078432
    '''

    return (np.log2(num_classes) +
            p_correct*np.log2(p_correct) +
            (1-p_correct)*np.log2((1-p_correct)/(num_classes-1)))/delta_t


class IterPlotter:
    @auto_attribute
    def __init__(
        self,
        plot_fxn,
        iterator_fxn,
    ):
        pass

    def _plot(self, iterate):
        display.clear_output(wait=True)
        self.plot_fxn(iterate)
        plt.show()

    def iter_plot(self):
        for iterate in self.iterator_fxn():
            self._plot(iterate)
            user_input = input("Press Enter to continue...")
            if user_input == 'q':
                print('quitting...')
                break


# for importing from json files.  Cribbed from ???
def str2int_hook(d):
    return {int(k) if k.lstrip('-').isdigit() else k: v for k, v in d.items()}


def anti_alias_and_resample(
    data, F_sampling, F_target, F_high=None, F_transition=None, atten_DB=40,
    ZSCORE=True
):
    ##############
    # Retire?
    ##############

    if F_high is None:
        # just set it to Nyquist
        F_high = F_target/2
    if F_transition is None:
        F_transition = 0.2*F_target

    data = anti_alias(data, F_sampling, F_high, F_transition, atten_DB)
    source_to_target_ratio = F_sampling/F_target
    # if source_to_target_ratio == F_sampling//F_target:
    #     print('integer downsampling...')
    #     return data[::int(source_to_target_ratio)]
    # else:
    return resample(data, source_to_target_ratio, ZSCORE)


def anti_alias(data, F_sampling, F_high, F_transition, atten_DB):

    F_nyquist = F_sampling/2

    # The fred harris rule of thumb:
    #   N = [(fs)/delta(f)]∗[atten(dB)/22]
    #   https://dsp.stackexchange.com/questions/37646
    # But you run filtfilt, which effectively doubles the order, so only
    #   need N/2
    N = (F_sampling/F_transition*atten_DB/22)/2
    # N = 2**int(np.ceil(np.log2(N)))  # use a power of 2
    N = 2*int(N//2) + 1  # use odd number

    # filter forwards and backwards
    desired = (1, 1, 0, 0)
    bands = (0, F_high, F_high+F_transition, F_nyquist)
    fir_firls = signal.firls(N, bands, desired, fs=F_sampling)
    data_anti_aliased = np.zeros_like(data)
    for iElectrode, raw_signal in enumerate(data.T):
        data_anti_aliased[:, iElectrode] = signal.filtfilt(
            fir_firls, 1, raw_signal)

    return data_anti_aliased


def resample(
    data, source_to_target_ratio, ZSCORE, resample_method='sinc_best',
    N_channels_max=128
):

    ######################
    # If downsampling by an integer, just anti-alias and subsample??
    ######################

    # 128 is the max for the underlying library
    N_channels_max = min(N_channels_max, 128)
    N_channels = data.shape[1]
    data_mat = None

    for i0 in np.arange(0, N_channels, N_channels_max):
        iF = np.min((i0+N_channels_max, N_channels))
        resampler = samplerate.Resampler(resample_method, channels=iF-i0)
        data_chunk = resampler.process(
            data[:, i0:iF], 1/source_to_target_ratio, end_of_input=True
        )
        data_mat = (
            data_chunk if data_mat is None else
            np.concatenate((data_mat, data_chunk), axis=1)
        )
    if ZSCORE:
        data_mat = zscore(data_mat)

    return data_mat


def generate_password(N=8, special_characters=None):

    # these are pretty common restrictions
    if special_characters is None:
        special_characters = '#&%!@()'

    # assemble all the required character types
    character_strings = [
        string.ascii_lowercase,
        string.ascii_uppercase,
        string.digits,
        special_characters,
    ]
    M = len(character_strings)

    # Get at least one of every kind of character, but making sure
    #  to get *in total* N characters
    nums_chars = np.random.multinomial(N-M, [1/M]*M) + np.array([1]*M)
    password = []
    for num_chars, character_string in zip(nums_chars, character_strings):
        password += list(
            np.random.choice(np.array(list(character_string)), num_chars)
        )

    return ''.join(list(np.random.choice(np.array(password), N, replace=False)))


def time2index(event_times, sampling_rate=None, analog_times=None):
    '''
    Find the index of the analog_time nearest the event_times, either by
    expliciting looking at the vector of analog_times, or (if it is uniformly
    sampled) simply by multiplying the event_times by the sampling_rate and
    rounding.
    '''

    # xor: one or the other but not both of the inputs must be None
    assert (sampling_rate is None) != (analog_times is None)

    # for uniformly sampled data, it's simple
    if sampling_rate is not None:
        event_indices = np.rint(sampling_rate*event_times).astype(int)
        return event_indices

    # for non-uniformly sampled (but still ordered) times it's more complicated
    else:        
        # Modified from https://stackoverflow.com/questions/2566412

        # find the index where we would insert the events (binary search)
        event_indices = np.searchsorted(analog_times, event_times, side="left")

        # events lying beyond last index are closest to last index
        BEYOND_LAST_INDEX = event_indices == len(analog_times)
        event_indices[BEYOND_LAST_INDEX] -= 1

        # events closer to the previous index
        CLOSER_TO_PREVIOUS_INDEX = (
            # distance to previous index (unless there is no prev. index)
            np.fabs(event_times - analog_times[np.maximum(
                event_indices-1, 0)]
            ) < 
            # distance to next index
            np.fabs(event_times - analog_times[event_indices])
        )
        event_indices[CLOSER_TO_PREVIOUS_INDEX] -= 1

        return event_indices
