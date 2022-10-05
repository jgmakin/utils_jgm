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
            self.figures, self.plots_dict, slider_updates = self.local_plotter(
                **{
                    key: slider.value for key, slider
                    in self.independent_sliders.items()
                },
            )

            # ...
            for slider in self.independent_sliders.values():
                slider.observe(self.update_response, 'value')

            # update the dependent sliders
            if hasattr(self, 'dependent_sliders'):
                # update dependent sliders
                for key, value in slider_updates.items():
                    self.dependent_sliders[key].value = value

        else:
            self.figures = []
            self.plots_dict = {}
            self.independent_sliders = {}

        if not hasattr(self, 'dependent_sliders'):
            self.dependent_sliders = {}

        self.sleep = False

    def update_response(self, change):
        # update plots
        _, _, slider_updates = self.local_plotter(
            **{key: slider.value for key, slider in self.independent_sliders.items()},
            plots_dict=self.plots_dict,
        )
        
        time.sleep(0.2)
        if hasattr(self, 'dependent_sliders'):
            # update dependent sliders
            for key, value in slider_updates.items():
                self.dependent_sliders[key].value = value

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
                _, _, slider_updates = self.local_plotter(
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
                if hasattr(self, 'dependent_sliders'):
                    for key, value in slider_updates.items():
                        self.dependent_sliders[key].value = value
                    
                # make the scan take 1 second
                if self.sleep:
                    time.sleep(self.sleep/len(scan_values))

    def get_layout(self):
        # the layout of plots and sliders
        layout = Layout(
            # flex='0 1 auto',
            height='auto',
            # min_height='4000px',
            width='auto',
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
        final_layout = HBox(
            [
                VBox(self.figures),
                VBox([slider_layout], layout=Layout(
                    # flex='0 1 auto',
                    height='auto',
                    #min_width='1000px',
                    width='1000px'  # 'auto'
                )),
            ],
            layout=layout,
        )

        return final_layout

    @staticmethod
    def local_plotter(**kwargs):
        pass

