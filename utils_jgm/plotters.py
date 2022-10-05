# standard libraries
import pdb

# third-party libraries
import numpy as np
from scipy.stats import multivariate_normal
from scipy import signal
import bqplot as bq
import bqplot.pyplot as bplt
from ipywidgets import FloatSlider

# local libraries
from utils_jgm.widgetizer import Widgetizer
from utils_jgm.toolbox import auto_attribute
from utils_jgm import tau




class NormalizingFlowWidgetizer(Widgetizer):
    def __init__(self):
        self.independent_sliders = {
            'b': FloatSlider(
                description='b', value=0, min=-15, max=15, step=0.25,
                readout_format='.2e', orientation='vertical',
            ),
            'w_angle': FloatSlider(
                description='w_angle', value=5*np.pi/4, min=0, max=2*np.pi,
                step=np.pi/16, readout_format='.2e', orientation='vertical',
            ),
            'w_mag': FloatSlider(
                description='w_mag', value=3, min=-5, max=5, step=0.25,
                readout_format='.2e', orientation='vertical',
            ),
            'u_angle': FloatSlider(
                description='u_angle', value=np.pi/2, min=0, max=2*np.pi,
                step=np.pi/16, readout_format='.2e', orientation='vertical',
            ),
            'u_mag': FloatSlider(
                description='u_mag', value=1, min=-5, max=5, step=0.25,
                readout_format='.2e', orientation='vertical',
            ),
        }
        
        super().__init__()

    @staticmethod
    def local_plotter(**kwargs):
        return planar_flow(**kwargs)


def planar_flow(
    b=0, w_angle=5*np.pi/4, w_mag=3, u_angle=np.pi/2, u_mag=1, plots_dict=None,
):

    # ...
    N = 500

    # 
    if plots_dict is None:
        plots_dict = dict().fromkeys(
            ['source data', 'flowed data', 'projection axis', 'translation axis']
        )

    # (re-)sample source data if the plot_data is None
    if plots_dict['source data'] is None:
        # random data (put elsewhere...)
        mu = np.array([0, 0])
        Sigma = np.array([[1, 0], [0, 1]])
        mvn = multivariate_normal(mu, Sigma)
        Z = mvn.rvs(N)
    else:
        Z = np.array([plots_dict['source data'].x, plots_dict['source data'].y]).T
    
    # polar->cartesian for projection and shift axes
    w = w_mag*np.array([np.cos(w_angle), np.sin(w_angle)])
    u = u_mag*np.array([np.cos(u_angle), np.sin(u_angle)])
    
    # the transformation
    Y = Z + u*np.tanh((Z@w[:, None] + b))
    
    # if all the plots are empty...
    if all(plot_data is None for plot_data in plots_dict.values()):

        # ...then (re-)populate a figure from scratch
        Ymin = Y.min(0)*2
        Ymax = Y.max(0)*2
        Yrange = Ymax - Ymin
        fig = bplt.figure(
            title='Planar flow',
            scales={
                'x': bq.LinearScale(min=Ymin[0], max=Ymax[0]),
                'y': bq.LinearScale(min=Ymin[1], max=Ymax[1]),
            },
            x_axis_label='Re',
            y_axis_label='Im',
            min_aspect_ratio=Yrange[0]/Yrange[1],
            max_aspect_ratio=Yrange[0]/Yrange[1],
        )
        plots_dict['source data'] = bplt.scatter(
            [], [], colors=['blue'], opacity=[0.1]*Z.shape[0]
        )
        plots_dict['flowed data'] = bplt.scatter(
            [], [], colors=['orange'], opacity=[0.1]*Y.shape[0]
        )
        plots_dict['projection axis'] = bplt.plot(
            [], [], 'r', stroke_width=6
        )
        plots_dict['translation axis'] = bplt.plot(
            [], [], 'g', stroke_width=6
        )

        figures = [fig]
    else:
        figures = []
        
    # just change the values on
    plots_dict['source data'].x = Z[:, 0]
    plots_dict['source data'].y = Z[:, 1]

    plots_dict['flowed data'].x = Y[:, 0]
    plots_dict['flowed data'].y = Y[:, 1]
    
    plots_dict['projection axis'].x = [0, w[0]]
    plots_dict['projection axis'].y = [0, w[1]]
    
    plots_dict['translation axis'].x = [0, u[0]]
    plots_dict['translation axis'].y = [0, u[1]]
    
    return figures, plots_dict, {}





class EulersWidgetizer(Widgetizer):
    def __init__(self):
        
        omega_slider = FloatSlider(
            description='omega', value=0.1, min=0.1, max=10, step=0.1,
            readout_format='.2e', orientation='vertical',
        )
        self.independent_sliders = {
            'omega': omega_slider,
            'phi': FloatSlider(
                description='phi', value=0, min=0, max=tau, step=tau/20,
                readout_format='.2e', orientation='vertical',
            ),
            't': FloatSlider(
                description='t', value=0, min=0, max=5*tau/omega_slider.min,
                step=tau/(omega_slider.min*20),
                readout_format='.2e', orientation='vertical',
            ),
        }
        super().__init__()

    @staticmethod
    def local_plotter(**kwargs):
        return rotating_complex_exponentials(**kwargs)


def rotating_complex_exponentials(
    omega=0.1, phi=0, t=0, plots_dict=None,
):

    # ...
    N = 100

    # 
    if plots_dict is None:
        plots_dict = dict().fromkeys(['counterclockwise', 'clockwise', 'average'])

    # ...
    # go back in time 1/4 of the circle
    t0 = t - tau/4/omega

    # if all the plots are empty...
    if all(plot_data is None for plot_data in plots_dict.values()):

        lin_x = bq.LinearScale()
        lin_y = bq.LinearScale()
        
        # ...then (re-)populate a figure from scratch
        tt = np.linspace(0, tau/omega, N)
        unit_circle = bq.Lines(
            x=np.cos(tt), y=np.sin(tt), colors=['black'],
            opacities=[0.1],
            scales={
                "x": lin_x,
                "y": lin_y,
            },
            labels=['unit circle'],
            display_legend=True,
        )
        plots_dict['counterclockwise'] = bq.FlexLine(
            x=np.zeros(N), y=np.zeros(N), color=np.linspace(0,1,N),
            scales={
                "x": lin_x,
                "y": lin_y,
                "color": bq.ColorScale(colors=["white", "yellow"])
            },
            stroke_width=10,
            labels=['forward'],
            display_legend=True,
        )
        plots_dict['clockwise'] = bq.FlexLine(
            x=np.zeros(N), y=np.zeros(N), color=np.linspace(0,1,N),
            scales={
                "x": lin_x,
                "y": lin_y,
                "color": bq.ColorScale(colors=["white", "blue"])
            },
            stroke_width=10,
            labels=['backward'],
            display_legend=True,
        )
        plots_dict['average'] = bq.FlexLine(
            x=np.zeros(N), y=np.zeros(N), color=np.linspace(0,1,N),
            scales={
                "x": lin_x,
                "y": lin_y,
                "color": bq.ColorScale(colors=["white", "green"])
            },
            stroke_width=10,
            display_legend=True,
            labels=['cos'],
        )
        
        z = 1.2
        fig = bplt.figure(
            title="Euler's identity",
            scales={
                'x': bq.LinearScale(min=-z, max=z),
                'y': bq.LinearScale(min=-z, max=z),
            },
            marks=list(plots_dict.values()) + [unit_circle],
            x_axis_label='Re',
            y_axis_label='Im',
            min_aspect_ratio=1,
            max_aspect_ratio=1,
        )
        # bplt.legend()
        figures = [fig]
    else:
        figures = []
     
    
    f1 = lambda t: np.exp(1j*(omega*t + phi))
    f2 = lambda t: np.exp(-1j*(omega*t + phi))
    f3 = lambda t: np.cos(omega*t + phi)

    # just change the values on
    plots_dict['counterclockwise'].x = np.real(f1(np.linspace(t0,t,N)))
    plots_dict['counterclockwise'].y = np.imag(f1(np.linspace(t0,t,N)))

    plots_dict['clockwise'].x = np.real(f2(np.linspace(t0,t,N)))
    plots_dict['clockwise'].y = np.imag(f2(np.linspace(t0,t,N)))
    
    plots_dict['average'].x = np.real(f3(np.linspace(t0,t,N)))
    plots_dict['average'].y = np.zeros(N)
    
    return figures, plots_dict, {}







class RLC_circuit():
    @auto_attribute
    def __init__(self, R, L, C, PARALLEL=True):
        
        # ...
        pass
    
    @property
    def w0(self):
        return 1/(self.L*self.C)**(1/2)
    
    @property
    def bandwidth(self):
        if self.PARALLEL:
            return 1/(self.R*self.C)
        else:
            return self.R/self.C
        
    @property
    def quality_factor(self):
        return self.w0/self.bandwidth
    
    @property
    def w_L(self):
        return -self.bandwidth/2 + ((self.bandwidth/2)**2 + self.w0**2)**(1/2)
    
    @property
    def w_H(self):
        return self.bandwidth/2 + ((self.bandwidth/2)**2 + self.w0**2)**(1/2)
        
    @property
    def poles(self):
        # the roots of the charactertistic polynomial
        gg = ((self.bandwidth/2)**2 - self.w0**2)**(1/2)
        w1 = +gg - self.bandwidth/2
        w2 = -gg - self.bandwidth/2
        return [w1, w2]

    def temporal_response(self, x0=0, xdot0=0, t_max=0.2):
        legend_str = []
        
        t = np.linspace(0,t_max,10000)
        
        s1, s2 = self.poles
        if s1 == s2:
            a = np.array([[x0], [xdot0 - s1*x0]])
            x = a[0]*np.e**(s1*t) + a[1]*t*np.e**(s2*t)
            legend_str.append('critically damped')
        else:
            if np.isreal(s1):
                a = np.array([
                    [s2, -1],
                    [-s1, 1],
                ])@np.array([[x0], [xdot0]])/(s2 - s1)
                x = a[0]*np.e**(s1*t) + a[1]*np.e**(s2*t)
                legend_str.append('overdamped')
            else:
                if s1==-s2:
                    w = abs(np.imag(s1))
                    a = np.array([[x0], [xdot0/w]])
                    x = a[0]*np.cos(w*t) + a[1]*np.sin(w*t)
                    legend_str.append('undamped')
                else:
                    w = abs(np.imag(s1))
                    alpha = np.real(s1)
                    a = np.array([[x0], [(xdot0 - alpha*x0)/w]])
                    x = np.e**(alpha*t)*(a[0]*np.cos(w*t) + a[1]*np.sin(w*t))
                    legend_str.append('underdamped')

        # print(R, a[0], a[1], s1, s2)
        return t, x

    def frequency_response(self, zeros=[0], gain=1, frequencies=None):
        ZPG_model = signal.ZerosPolesGain(zeros, self.poles, gain)
        frequencies, magnitude, phase = signal.bode(  #ZPG_model.bode()
            ZPG_model, w=frequencies 
        )
        
        return frequencies, magnitude, phase


#######
# consider adding zeros to circuit properties? and gain??
#######
def plot_temporal_response(circuit, x0, xdot0, t_max=0.2, line_t=None):
    # temporal response
    t, amplitude = circuit.temporal_response(x0, xdot0, t_max)
    
    if line_t is None:
        fig_t = bplt.figure()
        line_t = bplt.plot(t, amplitude, 'm', stroke_width=3)    
        return fig_t, line_t
    else:
        print('updating')
        line_t.x = t
        line_t.y = amplitude
    

def Bode_plot(
    circuit, zeros, gain=1,
    magnitude_plot=None, phase_plot=None, frequencies=None
):
    
    # Compute the maginitude and phase responses--at all frequencies but also
    #  again at the poles and zeros
    plots = {'magnitude': magnitude_plot, 'phase': phase_plot}
    titles = {'magnitude': '|H(j\u03C9)|', 'phase': '\u2220 H(j\u03C9)'}

    # superimpose each of the following plots
    plot_specs = {
        'all_frequencies': {
            'frequencies': frequencies,
            'responses': {'magnitude': None, 'phase': None},
            'phase_response': None,
            'plot_type': 'plot',
            'marker': None,
            'color': ['red']
        },
        'at poles': {
            'frequencies': [abs(pole) for pole in circuit.poles if abs(pole) > 0],
            'responses': {'magnitude': None, 'phase': None},
            'plot_type': 'scatter',
            'marker': 'cross',
            'color': ['green']
        },
        'at zeros': {
            'frequencies': [abs(zero) for zero in zeros if abs(zero) > 0],
            'responses': {'magnitude': None, 'phase': None},
            'plot_type': 'scatter',
            'marker': 'circle',
            'color': ['blue'],
        },
        'at corner frequencies': {
            'frequencies': [abs(w) for w in [circuit.w_L, circuit.w_H] if abs(w) > 0],
            'responses': {'magnitude': None, 'phase': None},
            'plot_type': 'scatter',
            'marker': 'diamond',
            'color': ['purple']
        }
    }

    # get all responses
    for spec in plot_specs.values():
        (
            spec['frequencies'],
            spec['responses']['magnitude'],
            spec['responses']['phase']
        ) = circuit.frequency_response(zeros, gain, spec['frequencies'])
        spec['responses']['phase'] *= np.pi/180  # convert to radians
    
    # ....
    scale_x = bq.LogScale()
    for key, plot in plots.items():
        if plot is None:
            plot = {}
            plot['figure'] = bplt.figure(
                title=titles[key],
                scales={'x': scale_x, 'y': bq.LinearScale()},
            )
            # fig_magnitude.axes = [
            #     bq.axes.Axis(
            #         label='\u03C9 (rad/s)',
            #         tick_format='0.2e',
            #         scale=scale_x,
            #     ),
            #     bq.axes.Axis(
            #         label='dB',
            #         tick_format='0.2f',
            #         orientation='vertical',
            #         scale=bq.LinearScale(),
            #     )
            # ]

            for spec_name, spec in plot_specs.items():
                try:
                    plot[spec_name] = getattr(bplt, spec['plot_type'])(
                        spec['frequencies'], spec['responses'][key],
                        marker=spec['marker'], colors=spec['color']
                    )
                except:
                    pdb.set_trace()

            # plot['corner-frequency scatter'] = bplt.scatter(
            #     zero_magnitudes,
            #     response_at_zeros[key],
            #     marker='circle'
            # )
        else:
            for spec_name, spec in plot_specs.items():
                plot[spec_name].x = spec['frequencies']
                plot[spec_name].y = spec['responses'][key]

                # plot['line'].y = response[key]
                # plot['pole scatter'].x = pole_magnitudes
                # plot['pole scatter'].y = response_at_poles[key]
                # plot['zero scatter'].x = zero_magnitudes
                # plot['zero scatter'].y = response_at_zeros[key]

        # update the plot
        plots[key] = plot
            
    return plots['magnitude'], plots['phase']

        
def pole_zero_plot(circuit, zeros, plot=None):

    # for scaling properly
    xmin = min([*np.real(circuit.poles), -circuit.w0])*1.1
    xmax = max([*np.real(circuit.poles), circuit.w0])*1.1
    ymin = min([*np.imag(circuit.poles), -circuit.w0])*1.1
    ymax = max([*np.imag(circuit.poles), circuit.w0])*1.1
    
    if plot is None:
        plot = {}

        # the unit circle
        plot['figure'] = bplt.figure(
            title='Pole-Zero plot',
            scales={
                'x': bq.LinearScale(min=xmin, max=xmax),
                'y': bq.LinearScale(min=ymin, max=ymax),
            },
            x_axis_label='Re',
            y_axis_label='Im',
            min_aspect_ratio=(xmax-xmin)/(ymax-ymin),
            max_aspect_ratio=(xmax-xmin)/(ymax-ymin),
            #plot_height=500,
            #tooltips=tooltips,
        )
        plot['unit circle'] = bplt.plot(
            *get_circle_data(circuit.w0),
            line_dash='dashed'
        )    
        plot['zeros'] = bplt.scatter(np.real(zeros), np.imag(zeros))
        plot['poles'] = bplt.scatter(
            np.real(circuit.poles),
            np.imag(circuit.poles),
            marker='cross'
        )
    else:
        plot['unit circle'].x, plot['unit circle'].y = get_circle_data(circuit.w0)
        plot['zeros'].x, plot['zeros'].y = np.real(zeros), np.imag(zeros)
        plot['poles'].x, plot['poles'].y = np.real(circuit.poles), np.imag(circuit.poles)
        #######
        # update aspect ratio
        #######
        
    return plot


def get_circle_data(w0):
    x = np.linspace(-w0, w0, 1000)
    y = (w0**2 - x**2)**(1/2)
    return (
        np.concatenate((x, np.flip(x))),
        np.concatenate((y, -np.flip(y)))
    )


#########
circuit = RLC_circuit(1, 1, 1)
frequencies = np.logspace(-3, 7, 1000)
t_max = 0.2
zeros = [0]
#########

class RLCWidgetizer(Widgetizer):
    def __init__(self):
        R = 50
        L = 1
        C = 100e-6

        #################
        self.independent_sliders = {
            'R': FloatSlider(
                description='R', value=R, min=0.5, max=1000, step=0.5,
                readout_format='.2e', orientation='vertical',),
            'L': FloatSlider(
                description='L', value=L, min=1/3, max=10, step=1/3,
                orientation='vertical'),
            'C': FloatSlider(
                description='C', value=C, min=10e-6, max=0.25, step=10e-6,
                readout_format='.2e', orientation='vertical',
            ),
            'x0': FloatSlider(
                description='$x_0$', value=0, min=0, max=10, step=0.1,
                orientation='vertical',),
            'xdot0': FloatSlider(
                description='$\dot x_0$', value=1000, min=0, max=10000,
                step=100, orientation='vertical',
            ),
        }
        #################

        self.dependent_sliders = {
            'w0': FloatSlider(
                description='$\omega_0$',
                value=1/(self.independent_sliders['R'].value*self.independent_sliders['C'].value)**(1/2),
                min=1/(self.independent_sliders['L'].max*self.independent_sliders['C'].max)**(1/2),
                max=1/(self.independent_sliders['L'].min*self.independent_sliders['C'].min)**(1/2),
                orientation='vertical',
                step=5
            ),
            'bandwidth': FloatSlider(
                description='$B_{\omega}$',
                value=1/(self.independent_sliders['R'].value*self.independent_sliders['C'].value),
                min=1/(self.independent_sliders['R'].max*self.independent_sliders['C'].max),
                max=1/(self.independent_sliders['R'].min*self.independent_sliders['C'].min),
                orientation='vertical',
                step=5
            )
        }
        self.dependent_sliders = {
            **self.dependent_sliders,
            'quality_factor': FloatSlider(
                description='$Q$',
                value=self.dependent_sliders['w0'].value/self.dependent_sliders['bandwidth'].value,
                min=self.dependent_sliders['w0'].min/self.dependent_sliders['bandwidth'].max,
                max=self.dependent_sliders['w0'].max/self.dependent_sliders['bandwidth'].min,
                orientation='vertical',
                step=5
            ),
            'w_L': FloatSlider(
                description='$\omega_L$',
                value=-self.dependent_sliders['bandwidth'].value/2 + (
                    (self.dependent_sliders['bandwidth'].value/2)**2 + 
                    self.dependent_sliders['w0'].value**2)**(1/2),
                min=-self.dependent_sliders['bandwidth'].max/2 + (
                    (self.dependent_sliders['bandwidth'].max/2)**2 +
                    self.dependent_sliders['w0'].min**2)**(1/2),
                max=-self.dependent_sliders['bandwidth'].min/2 + (
                    (self.dependent_sliders['bandwidth'].max/2)**2 +
                    self.dependent_sliders['w0'].max**2)**(1/2),
                orientation='vertical',
                step=5
            ),
            'w_H': FloatSlider(
                description='$\omega_H$',
                value=self.dependent_sliders['bandwidth'].value/2 + (
                    (self.dependent_sliders['bandwidth'].value/2)**2 +
                    self.dependent_sliders['w0'].value**2)**(1/2),
                min=self.dependent_sliders['bandwidth'].min/2 + (
                    (self.dependent_sliders['bandwidth'].min/2)**2 +
                    self.dependent_sliders['w0'].min**2)**(1/2),
                max=self.dependent_sliders['bandwidth'].max/2 + (
                    (self.dependent_sliders['bandwidth'].max/2)**2 +
                    self.dependent_sliders['w0'].max**2)**(1/2),
                orientation='vertical',
                step=5
            ),
            'p1': FloatSlider(
                description='$p_1$',
                #value=circuit.poles[0],
                value=-self.dependent_sliders['bandwidth'].value/2 - (
                    (self.dependent_sliders['bandwidth'].value/2)**2 -
                    self.dependent_sliders['w0'].value**2)**(1/2),
                # min=-self.dependent_sliders['bandwidth'].max/2 - (
                #     (self.dependent_sliders['bandwidth'].max/2)**2 -
                #     self.dependent_sliders['w0'].min)**(1/2),
                # max=-self.dependent_sliders['bandwidth'].value/2 - (
                #     (self.dependent_sliders['bandwidth'].value/2)**2 -
                #     self.dependent_sliders['w0'].value)**(1/2),
                orientation='vertical',
            )
        }
        
        super().__init__()

    @staticmethod
    def local_plotter(**kwargs):
        return RLC_plotter(**kwargs)


def RLC_plotter(
    R=1, L=1, C=1, x0=0, xdot0=0, plots_dict=None,
):
    if plots_dict is None:
        plots_dict = dict().fromkeys(
            ['temporal', 'Bode magnitude', 'Bode phase', 'pole-zero']
        )

    # update the circuit
    circuit.R = R
    circuit.L = L
    circuit.C = C

    # temporal response
    figures = []
    plot_info = plot_temporal_response(
        circuit, x0, xdot0, t_max, plots_dict['temporal']
    )
    if plot_info is not None:
        fig, plots_dict['temporal'] = plot_info
        figures.append(fig)

    # pole-zero plot
    plots_dict['pole-zero'] = pole_zero_plot(
        circuit, zeros, plots_dict['pole-zero']
    )
    if plots_dict['pole-zero']['figure'] is not None:
        # ...and it never should be...
        figures.append(plots_dict['pole-zero']['figure'])

    # Bode plots
    plot_info = Bode_plot(
        circuit, zeros, 1/circuit.C, plots_dict['Bode magnitude'],
        plots_dict['Bode phase'],
        frequencies=frequencies,
    )
    if plot_info is not None:
        plots_dict['Bode magnitude'], plots_dict['Bode phase'] = plot_info
        figures += [
            plots_dict['Bode magnitude']['figure'],
            plots_dict['Bode phase']['figure'],
        ]

    # # update sliders that are dependent on other sliders
    slider_updates = {}
    for key in ['w0', 'bandwidth', 'quality_factor', 'w_L', 'w_H']:
        slider_updates[key] = getattr(circuit, key)

    #####
    # the slider display doesn't like when this is complex.  May be able to fix
    #  with the readout_format
    # slider_updates['p1'] = getattr(circuit, 'poles')[0]
    #####

    return figures, plots_dict, slider_updates







class ConvolutionWidgetizer(Widgetizer):
    def __init__(self):
        R = 50
        L = 1
        C = 100e-6

        #################
        self.independent_sliders = {
            'R': FloatSlider(
                description='R', value=R, min=0.5, max=1000, step=0.5,
                readout_format='.2e', orientation='vertical',),
            'L': FloatSlider(
                description='L', value=L, min=1/3, max=10, step=1/3,
                orientation='vertical'),
            'C': FloatSlider(
                description='C', value=C, min=10e-6, max=0.25, step=10e-6,
                readout_format='.2e', orientation='vertical',
            ),
            'x0': FloatSlider(
                description='$x_0$', value=0, min=0, max=10, step=0.1,
                orientation='vertical',),
            'xdot0': FloatSlider(
                description='$\dot x_0$', value=1000, min=0, max=10000,
                step=100, orientation='vertical',
            ),
        }
        super().__init__()

    @staticmethod
    def local_plotter(**kwargs):
        return convolution_plotter(**kwargs)


def convolution_plotter(
    R=1, L=1, C=1, x0=0, xdot0=0, plots_dict=None,
):
    if plots_dict is None:
        plots_dict = dict().fromkeys(
            ['temporal', 'Bode magnitude', 'Bode phase', 'pole-zero']
        )

    # update the circuit
    circuit.R = R
    circuit.L = L
    circuit.C = C

    # temporal response
    figures = []
    plot_info = plot_temporal_response(
        circuit, x0, xdot0, t_max, plots_dict['temporal']
    )
    if plot_info is not None:
        fig, plots_dict['temporal'] = plot_info
        figures.append(fig)

    # pole-zero plot
    plots_dict['pole-zero'] = pole_zero_plot(
        circuit, zeros, plots_dict['pole-zero']
    )
    if plots_dict['pole-zero']['figure'] is not None:
        # ...and it never should be...
        figures.append(plots_dict['pole-zero']['figure'])

    # Bode plots
    plot_info = Bode_plot(
        circuit, zeros, 1/circuit.C, plots_dict['Bode magnitude'],
        plots_dict['Bode phase'],
        frequencies=frequencies,
    )
    if plot_info is not None:
        plots_dict['Bode magnitude'], plots_dict['Bode phase'] = plot_info
        figures += [
            plots_dict['Bode magnitude']['figure'],
            plots_dict['Bode phase']['figure'],
        ]

    # # update sliders that are dependent on other sliders
    slider_updates = {}
    for key in ['w0', 'bandwidth', 'quality_factor', 'w_L', 'w_H']:
        slider_updates[key] = getattr(circuit, key)

    #####
    # the slider display doesn't like when this is complex.  May be able to fix
    #  with the readout_format
    # slider_updates['p1'] = getattr(circuit, 'poles')[0]
    #####

    return figures, plots_dict, slider_updates
