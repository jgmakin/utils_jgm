# standard libraries
from functools import reduce
import pdb
import codecs
import six

# third-party packages
import tikzplotlib as tpl


def tpl_save(*args, **kwargs):

    # these are not passed to get_tikz_code
    encoding = kwargs.pop("encoding", None)
    extra_body_parameters = kwargs.pop('extra_body_parameters', None)

    # always pass certain tikzpicture parameters
    extra_tikzpicture_parameters = kwargs.pop(
        'extra_tikzpicture_parameters', None)
    standard_tikzpicture_parameters = {
        '\\providecommand{\\thisXlabelopacity}{1.0}',
        '\\providecommand{\\thisYlabelopacity}{1.0}',
        '\\pgfplotsset{compat=1.15}',
    }
    extra_tikzpicture_parameters = (
        standard_tikzpicture_parameters
        if extra_tikzpicture_parameters is None else
        extra_tikzpicture_parameters | standard_tikzpicture_parameters
    )

    # always pass certain axis parameters
    extra_axis_parameters = kwargs.pop('extra_axis_parameters', None)
    standard_axis_parameters = {
        'every axis x label/.append style={opacity=\\thisXlabelopacity}',
        'every axis y label/.append style={opacity=\\thisYlabelopacity}',
    }
    extra_axis_parameters = (
        standard_axis_parameters if extra_axis_parameters is None else
        extra_axis_parameters | standard_axis_parameters
    )

    # get the code
    code = tpl.get_tikz_code(
        *args, **kwargs,
        extra_axis_parameters=extra_axis_parameters,
        extra_tikzpicture_parameters=extra_tikzpicture_parameters,
    )

    # perhaps tack some extra code before the \end{axis} command
    if extra_body_parameters is not None:
        end_axis = '\\end{axis}'
        code = tpl.get_tikz_code(
            *args, **kwargs,
            extra_axis_parameters=extra_axis_parameters,
            extra_tikzpicture_parameters=extra_tikzpicture_parameters,
        )
        code_pieces = code.split(end_axis)
        code = (
            code_pieces[0] +
            reduce(lambda a, b: b+'\n'+a, reversed(extra_body_parameters), end_axis) +
            code_pieces[1]
        )

    # ...
    filepath = kwargs.pop('filepath')
    file_handle = codecs.open(filepath, "w", encoding)
    try:
        file_handle.write(code)
    except UnicodeEncodeError:
        # We're probably using Python 2, so treat unicode explicitly
        file_handle.write(six.text_type(code).encode("utf-8"))
    file_handle.close()
