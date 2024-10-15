# standard libraries
import time
from functools import partial
import pdb

# third-party libraries
import numpy as np
from ipywidgets import HBox, VBox, Layout, Button


class Widgetizer():
    def __init__(
        self,
    ):
        # initialize
        if hasattr(self, 'independent_sliders'):
            # run once to get outputs (?)
            self.figures, self.plots_dict = self.local_plotter(
                **{
                    key: slider.value for key, slider
                    in self.independent_sliders.items()
                },
            )

            # make each slider update_response whenever it observes changes
            for slider in self.independent_sliders.values():
                slider.observe(self.update_response, 'value')

        else:
            self.figures = []
            self.plots_dict = {}
            self.independent_sliders = {}

        if not hasattr(self, 'dependent_sliders'):
            self.dependent_sliders = {}

        self.sleep = False

    def update_response(self, change):
        # update plots
        self.local_plotter(
            **{
                key: slider.value for key, slider
                in self.independent_sliders.items()
            },
            plots_dict=self.plots_dict,
        )
        
        time.sleep(0.2)
        self.set_slider_values()

    def scan_parameter(self, btn, param='b'):
        scan_values = np.arange(
            self.independent_sliders[param].min,
            self.independent_sliders[param].max,
            self.independent_sliders[param].step
        )
        for scan_value in scan_values:
            #############
            # hard-coded figures[0]---seems to work, though
            #############
            with self.figures[0].hold_sync():
                self.local_plotter(
                    **{
                        # update according to current slider values
                        **{
                            key: slider.value for key, slider
                            in self.independent_sliders.items()
                        },
                        # but overwrite with scanned parameter
                        **{param: scan_value},
                    },
                    plots_dict=self.plots_dict
                )

                # update the position of the slider being scanned
                self.independent_sliders[param].value = scan_value

                # update the dependent sliders
                self.set_slider_values()
                    
                # make the scan take 1 second
                if self.sleep:
                    time.sleep(self.sleep/len(scan_values))

    def get_layout(self):
        # the layout of plots and sliders
        layout = Layout(
            # flex='0 1 auto',
            height='auto',
            # min_height='4000px',
            # width='auto',
            width='4000px',
        )

        # create buttons for scanning the space
        buttons = {}
        for key in self.independent_sliders:
            buttons[key] = Button(icon="play")
            buttons[key].on_click(partial(self.scan_parameter, param=key))
            buttons[key].layout.width = '70px'

        # put together sliders and buttons
        slider_layout = VBox([
            HBox([
                VBox([self.independent_sliders[key], buttons[key]])
                for key in self.independent_sliders
            ]),
            HBox(list(self.dependent_sliders.values())),
            # HBox([sliders['p1']]),
        ])

        # put figure together with sliders
        final_layout = VBox(
            [
                HBox(self.figures),
                VBox(
                    [slider_layout],
                    layout=Layout(
                        # flex='0 1 auto',
                        height='auto',
                        #min_width='1000px',
                        width='1000px'  # 'auto'
                    )
                ),
            ],
            layout=layout,
        )

        return final_layout

    # shell
    def set_slider_values(self):
        pass

    @staticmethod
    def local_plotter(**kwargs):
        pass

