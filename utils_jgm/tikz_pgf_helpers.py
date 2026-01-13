# standard libraries
from functools import reduce
import pdb
import codecs
import six
import time

# third-party packages
# import tikzplotlib as tpl
import matplot2tikz as tpl


def tpl_save(
    #####
    # non-tpl args
    filepath,
    #####
    *args,
    #####
    # non-tpl kwargs
    encoding=None,
    extra_body_parameters=None,
    extra_groupstyle_parameters=None,
    #####
    #####
    # these kwargs have non-empty defaults (set below)
    extra_axis_parameters=None,
    extra_lines_start=None,
    #####
    **kwargs
):

    # (1) STANDARD PRE-TIKZPICTURE LINES (before everything)
    default_extra_lines_start = {
        '\\pgfplotsset{compat=1.18}%',
        '\\providecommand{\\figwidth}{360pt}%',
        '\\providecommand{\\figheight}{310pt}%',
        '\\providecommand{\\thisXlabelopacity}{1.0}%',
        '\\providecommand{\\thisYlabelopacity}{1.0}%',
    }

    # To allow the boolean to shut these params off, even if they had been
    #  explicitly set to values, use post/.append style
    for boolean, axis_param in zip(
        [
            'CLEANXAXIS', 'CLEANXAXIS', 'CLEANYAXIS', 'CLEANYAXIS',
            'CLEANTITLE', 'CLEARXLABEL', 'CLEARYLABEL',
        ],
        [
            'xticklabels', 'xlabel', 'yticklabels', 'ylabel',
            'title', 'xlabel', 'ylabel'
        ]
    ):
        default_extra_lines_start |= {
            '\\provideboolean{%s}'
            '\\ifthenelse{\\boolean{%s}}{%%'
            '\n\t\\pgfplotsset{every axis post/.append style={%s = {} }}%%'
            '\n}{}%%' % (boolean, boolean, axis_param)
        }

    # booleans for the extra_body_parameters
    extra_body_booleans = ['NOLEGEND']
    extra_body_commands = ['\\legend{}']
    for boolean in extra_body_booleans:
        default_extra_lines_start |= {
            '\\provideboolean{%s}%%' % (boolean)
        }

    # (2) STANDARD AXIS PARAMETERS (passed to \begin{axis})
    default_axis_parameters = {
        'width=\\figwidth',
        'height=\\figheight',
        'every axis x label/.append style={opacity=\\thisXlabelopacity}',
        'every axis y label/.append style={opacity=\\thisYlabelopacity}',
    }

    # (3) STANDARD BODY PARAMETERS (before \end{axis})
    for boolean, command in zip(
        extra_body_booleans, extra_body_commands
    ):
        default_body_parameters = {
            '\\ifthenelse{\\boolean{%s}}{%s}{}' % (boolean, command)
        }

    # now add in the extra content provided by the user as args
    # for lines, default_lines in zip(
    #     [extra_lines_start, extra_axis_parameters, extra_body_parameters],
    #     [default_extra_lines_start, default_axis_parameters,
    #      default_body_parameters]
    # ):
    #     lines = augment_params_set(lines, default_lines)
    extra_lines_start = augment_params(
        extra_lines_start, default_extra_lines_start)
    extra_axis_parameters = augment_params(
        extra_axis_parameters, default_axis_parameters)
    extra_body_parameters = augment_params(
        extra_body_parameters, default_body_parameters)

    # get the code, passing extra_axis_parameters
    code = tpl.get_tikz_code(
        filepath=filepath,  # need this for storing, e.g., png files
        *args, **kwargs,
        extra_axis_parameters=extra_axis_parameters,
        extra_groupstyle_parameters=extra_groupstyle_parameters,
        extra_lines_start=extra_lines_start,
    )

    # tack on code before axis is closed
    if extra_body_parameters is not None:
        end_axis = '\\end{axis}'
        code_pieces = code.split(end_axis)
        if len(code_pieces) > 1:
            code = (
                code_pieces[0] +
                reduce(lambda a, b: b+'\n'+a, reversed(extra_body_parameters), end_axis) +
                code_pieces[1]
            )
        else:
            print('uh oh: skipping extra_body_parameters')

    # add today's date to the top of the file
    code = '% ' + time.ctime() + '\n' + code

    # finally, write out the file
    file_handle = codecs.open(filepath, "w", encoding)
    try:
        file_handle.write(code)
    except UnicodeEncodeError:
        # We're probably using Python 2, so treat unicode explicitly
        file_handle.write(six.text_type(code).encode("utf-8"))
    file_handle.close()


def augment_params(passed, defaults):
    if type(passed) is set:
        return list(passed | defaults)
    elif passed is None:
        return list(defaults)
    elif type(passed) is list:
        return list(defaults) + passed
    else:
        raise TypeError('Expected list, set, or None')

    #return defaults if passed is None else passed | defaults


# https://github.com/nschloe/tikzplotlib/issues/557
def tikzplotlib_fix_ncols(obj):
    """
    workaround for matplotlib 3.6 renamed legend's _ncol to _ncols, which breaks tikzplotlib
    """
    if hasattr(obj, "_ncols"):
        obj._ncol = obj._ncols
    for child in obj.get_children():
        tikzplotlib_fix_ncols(child)
