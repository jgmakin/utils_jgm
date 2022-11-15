# standard libraries
from functools import reduce
import pdb
import codecs
import six

# third-party packages
import tikzplotlib as tpl


def tpl_save(
    #####
    # non-tpl args
    filepath,
    #####
    *args,
    #####
    # non-tpl kwargs
    encoding=None,
    pre_tikzpicture_lines=None,
    extra_body_parameters=None,
    #####
    #####
    # this kwarg has a non-empty default (set below)
    extra_axis_parameters=None,
    #####
    **kwargs
):

    # Always pass certain pre-tikzpicture lines.
    standard_pre_tikzpicture_lines = {
        '\\providecommand{\\thisXlabelopacity}{1.0}',
        '\\providecommand{\\thisYlabelopacity}{1.0}',
        '\\pgfplotsset{compat=1.15}',
    }

    # To allow the boolean to shut these params off, even if they had been
    #  explicitly set to values, use post/.append style
    for boolean, axis_param in zip(
        ['CLEANXAXIS', 'CLEANXAXIS', 'CLEANYAXIS', 'CLEANYAXIS'],
        ['xticklabels', 'xlabel', 'yticklabels', 'ylabel']
    ):
        standard_pre_tikzpicture_lines |= {
            '\\provideboolean{%s}'
            '\\ifthenelse{\\boolean{%s}}{%%'
            '\n\t\\pgfplotsset{every axis post/.append style={%s = {} }}%%'
            '\n}{}%%' % (boolean, boolean, axis_param)
        }

    # don't forget any pre_tikzpicture_lines that have been passed as args
    pre_tikzpicture_lines = augment_params_set(
        pre_tikzpicture_lines, standard_pre_tikzpicture_lines
    )

    # always pass certain axis parameters
    standard_axis_parameters = {
        'every axis x label/.append style={opacity=\\thisXlabelopacity}',
        'every axis y label/.append style={opacity=\\thisYlabelopacity}',
    }
    extra_axis_parameters = augment_params_set(
        extra_axis_parameters, standard_axis_parameters
    )

    # get the code
    code = tpl.get_tikz_code(
        filepath=filepath,  # need this for storing, e.g., png files
        *args, **kwargs, extra_axis_parameters=extra_axis_parameters,
    )

    # tack some extra code before anything
    if pre_tikzpicture_lines is not None:
        code = '%\n'.join(pre_tikzpicture_lines) + '%\n' + code

    # perhaps tack some extra code before the \end{axis} command
    if extra_body_parameters is not None:
        end_axis = '\\end{axis}'
        code_pieces = code.split(end_axis)
        code = (
            code_pieces[0] +
            reduce(lambda a, b: b+'\n'+a, reversed(extra_body_parameters), end_axis) +
            code_pieces[1]
        )

    # finally, write out the file
    file_handle = codecs.open(filepath, "w", encoding)
    try:
        file_handle.write(code)
    except UnicodeEncodeError:
        # We're probably using Python 2, so treat unicode explicitly
        file_handle.write(six.text_type(code).encode("utf-8"))
    file_handle.close()


def augment_params_set(passed, defaults):
    return defaults if passed is None else passed | defaults
