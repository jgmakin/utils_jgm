# third-party libraries
from ipywidgets import HBox, VBox, Layout


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

            ###########
            # CURRENTLY BROKEN
            if hasattr(self, 'dependent_sliders'):
                # update dependent sliders
                for key, value in slider_updates.items():
                    self.dependent_sliders[key].value = value
            ###########

        else:
            self.figures = []
            self.plots_dict = {}
            self.independent_sliders = {}

        if not hasattr(self, 'dependent_sliders'):
            self.dependent_sliders = {}

    def update_response(self, change):
        # update plots
        self.local_plotter(
            **{key: slider.value for key, slider in self.independent_sliders.items()},
            plots_dict=self.plots_dict,
        )
        
        # # update sliders that are dependent on other sliders
        # for key in dependent_vars:
        #     sliders[key].value = getattr(circuit, key)
        # sliders['p1'].value = getattr(circuit, 'poles')[0]

    def get_layout(self):
        # the layout of plots and sliders
        layout = Layout(
            # flex='0 1 auto',
            height='auto',
            # min_height='4000px',
            width='auto'
        )

        slider_layout = VBox([
            HBox([slider for slider in self.independent_sliders.values()]),
            HBox([slider for slider in self.dependent_sliders.values()]),
            # HBox([sliders['p1']]),
        ])

        final_layout = HBox([
            VBox(self.figures),
            VBox([slider_layout], layout=Layout(
                # flex='0 1 auto',
                height='auto',
                #min_width='1000px',
                width='1000px'  # 'auto'
            )),
        ], layout=layout)

        return final_layout

    @staticmethod
    def local_plotter(**kwargs):
        pass

