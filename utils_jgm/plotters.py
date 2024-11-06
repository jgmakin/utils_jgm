# standard libraries
import pdb

# third-party libraries
import numpy as np
from scipy.stats import multivariate_normal
from scipy import signal
import bqplot as bq
import bqplot.pyplot as bplt
from ipywidgets import FloatSlider, IntSlider

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
    
    return figures, plots_dict


class FunctionWidgetizer(Widgetizer):
    def __init__(self):
        self.independent_sliders = {
            'a': FloatSlider(
                description='a', value=0, min=-5, max=5, step=0.1,
                readout_format='.2e', orientation='vertical',
            ),
            'b': FloatSlider(
                description='b', value=0, min=-500, max=500, step=10,
                readout_format='.2e', orientation='vertical',
            ),
            'c': FloatSlider(
                description='c', value=1, min=-5, max=5, step=0.1,
                readout_format='.2e', orientation='vertical',
            ),
            'd': FloatSlider(
                description='d', value=1, min=-5, max=5, step=0.1,
                readout_format='.2e', orientation='vertical',
            ),
        }
        super().__init__()

    @staticmethod
    def local_plotter(**kwargs):
        return cubic_monomial(**kwargs)


def cubic_monomial(a=0, b=0, c=1, d=1, plots_dict=None):

    # init
    N = 1000
    t_min = -10
    t_max = 10
    t = np.linspace(t_min, t_max, N)

    def yr_func(aa, bb, cc, dd, tt):
        return dd*(cc*tt + aa)**3 + bb
        # return dd*(cc*tt + aa)**3 + bb

    x = yr_func(a, b, c, d, t)
    x_min = yr_func(0, 0, 1, 1, t_min)
    x_max = yr_func(0, 0, 1, 1, t_max)

    # first time in
    if plots_dict is None:
        plots_dict = dict().fromkeys(['cubic monomial'])

    # ...
    if all(plot_data is None for plot_data in plots_dict.values()):
        fig = bplt.figure(
            title="f(t) = d(ct + a)^3 + b",
            scales={
                'x': bq.LinearScale(min=t_min, max=t_max),
                'y': bq.LinearScale(min=x_min, max=x_max),
            },
        )
        plots_dict['cubic monomial'] = bplt.plot(t, x, 'm', stroke_width=3)
        
        # fig.layout.width = '600px'
        # bplt.legend()
        figures = [fig]
    else:
        figures = []

    plots_dict['cubic monomial'].x = t
    plots_dict['cubic monomial'].y = x

    return figures, plots_dict


class MOSFETAmplifierWidgetizer(Widgetizer):
    def __init__(self):
        self.independent_sliders = {
            'vgs_mag': FloatSlider(
                description='|vgs|', value=3, min=0.1, max=5, step=0.1,
                readout_format='.2e', orientation='vertical',
            ),
            'Q_VGS': FloatSlider(
                description='Q_VGS', value=0, min=-10, max=10, step=0.1,
                readout_format='.2e', orientation='vertical',
            ),
            'vT': FloatSlider(
                description='vT', value=0, min=-4, max=4, step=0.1,
                readout_format='.2e', orientation='vertical',
            ),
            'NMOSFET': IntSlider(
                description='NMOSFET', value=1, min=0, max=1, step=1,
                readout_format='d', orientation='vertical',
            ),
        }
        super().__init__()

    @staticmethod
    def local_plotter(**kwargs):
        return vds_vgs(**kwargs)


def vds_vgs(
    vgs_mag=3, Q_VGS=None, vT=0, NMOSFET=1, CLM=0.0, Rd=0.5, Rss=0.5, R1=500,
    R2=500, plots_dict=None
):

    # init
    NMOSFET = bool(NMOSFET)
    N = 1000
    k = 2
    vGS_max = 10
    vPlus = 5
    vMinus = -5
    Nperiods = 5

    # ...
    vDS_max = vPlus - vMinus
    iD_max = vDS_max/(Rd + Rss)

    if not NMOSFET:
        vGS_max = -vGS_max
        vDS_max = -vDS_max

    # range of vGS
    vGS = np.linspace(0, vGS_max, N)
    vDS, colors = get_MOSFET_voltage(vGS, vT, vDS_max, iD_max, k, N, NMOSFET)

    # ... 
    if Q_VGS is None:
        # compute all possible values of vG
        iD, _ = get_MOSFET_current(vGS, vDS, vT, k, N, NMOSFET, CLM)
        vS = vMinus + iD*Rss if NMOSFET else vPlus - iD*Rss
        vG = vS + vGS

        # find the one closest to that given by the gate-side resistors
        vG_actual = R2/(R1+R2)*vDS_max + vMinus
        Q_VGS = vGS[np.argmin(abs(vG_actual - vG))]

    # the quiescent point
    Q_VGS = np.ones(1)*Q_VGS
    Q_VDS, _ = get_MOSFET_voltage(Q_VGS, vT, vDS_max, iD_max, k, N, NMOSFET)
    
    # the input and output as a function of time
    t = np.linspace(0, Nperiods*tau, N)
    vGS_of_t = vgs_mag*np.sin(t) + Q_VGS
    vDS_of_t, _ = get_MOSFET_voltage(vGS_of_t, vT, vDS_max, iD_max, k, N, NMOSFET)

    # first time in
    if plots_dict is None:
        plots_dict = dict().fromkeys(['vDS vs. vGS'])
        plots_dict = dict().fromkeys(['Q point'])
        plots_dict = dict().fromkeys(['input'])
        plots_dict = dict().fromkeys(['output'])

    # ...
    if all(plot_data is None for plot_data in plots_dict.values()):
        
        lin_x = bq.LinearScale()
        lin_y = bq.LinearScale()

        # plot
        plots_dict['vDS vs. vGS'] = bq.FlexLine(
            x=np.zeros(N), y=np.zeros(N),
            color=colors,
            scales={
                "x": lin_x,
                "y": lin_y,
                "color": bq.OrdinalColorScale(scheme='Dark2', domain=[0, 1, 2])
            },
            stroke_width=4,
        )
        plots_dict['Q point'] = bplt.scatter(
            Q_VGS, Q_VDS, marker='circle',
            scales={
                "x": lin_x,
                "y": lin_y,
            },
        )

        ax_x = bplt.Axis(
            label='v_{GS}', scale=lin_x, grid_lines='solid', tick_format='0.2f'
        )
        ax_y = bplt.Axis(
            label='v_{DS}', scale=lin_y, orientation='vertical',
            tick_format='0.2f'
        )
        fig_amp = bplt.figure(
            title="MOSFET amplification",
            axes=[ax_x, ax_y],
            scales={
                'x': bq.LinearScale(min=0, max=8),
                'y': bq.LinearScale(min=0, max=15),
            },
            # marks=list(plots_dict.values()),
            marks=[
                plots_dict['vDS vs. vGS'],
                plots_dict['Q point'],
            ]
        )
        fig_amp.layout.width = '600px'
        # bplt.legend()

        # input and output
        fig_input = bplt.figure()
        fig_input.layout.width = '600px'
        fig_input.layout.height = '300px'
        plots_dict['input'] = bplt.plot(
            t, vGS_of_t,
            axes_options={'x': {"label": "t"}, "y": {"label": "v_{GS}"}}
        )

        fig_output = bplt.figure()
        fig_output.layout.width = '600px'
        fig_output.layout.height = '300px'
        plots_dict['output'] = bplt.plot(
            t, vDS_of_t,
            axes_options={'x': {"label": "t"}, "y": {"label": "v_{DS}"}}
        )

        figures = [fig_amp, fig_input, fig_output]
    else:
        figures = []

    plots_dict['vDS vs. vGS'].x = vGS
    plots_dict['vDS vs. vGS'].y = vDS
    plots_dict['vDS vs. vGS'].color = colors
    
    plots_dict['Q point'].x = Q_VGS
    plots_dict['Q point'].y = Q_VDS

    plots_dict['input'].y = vGS_of_t
    plots_dict['output'].y = vDS_of_t

    return figures, plots_dict


class MOSFETBiasCircuit(Widgetizer):
    def __init__(self):
        self.independent_sliders = {
            'R1': FloatSlider(
                description='R1', value=500, min=500, max=5000, step=10,
                readout_format='.2e', orientation='vertical',
            ),
            'R2': FloatSlider(
                description='R2', value=500, min=500, max=5000, step=10,
                readout_format='.2e', orientation='vertical',
            ),
            'Rd': FloatSlider(
                description='Rd', value=0.5, min=0.5, max=100, step=0.5,
                readout_format='.2e', orientation='vertical',
            ),
            'Rss': FloatSlider(
                description='Rss', value=0.5, min=0.5, max=100, step=0.5,
                readout_format='.2e', orientation='vertical',
            ),
            'vT': FloatSlider(
                description='vT', value=0, min=-4, max=4, step=0.1,
                readout_format='.2e', orientation='vertical',
            ),
            'NMOSFET': IntSlider(
                description='NMOSFET', value=1, min=0, max=1, step=1,
                readout_format='d', orientation='vertical',
            ),
        }
        super().__init__()

    @staticmethod
    def local_plotter(**kwargs):
        figures, plots_dict, _ = vds_vgs(**kwargs)
        plots_dict.pop('input')
        plots_dict.pop('output')
        return figures[0:1], plots_dict


class MOSFETOutputWidgetizer(Widgetizer):
    def __init__(self):
        self.independent_sliders = {
            'vGS': FloatSlider(
                description='vGS', value=3, min=-5, max=5, step=0.1,
                readout_format='.2e', orientation='vertical',
            ),
            'vT': FloatSlider(
                description='vT', value=2, min=-4, max=4, step=0.1,
                readout_format='.2e', orientation='vertical',
            ),
            'NMOSFET': IntSlider(
                description='NMOSFET', value=1, min=0, max=1, step=1,
                readout_format='d', orientation='vertical',
            ),
            'CLM': FloatSlider(
                description='CLM', value=0.0, min=0.0, max=0.02, step=0.002,
                readout_format='.2e', orientation='vertical',
            ),
        }
        super().__init__()

    @staticmethod
    def local_plotter(**kwargs):
        return id_vds(**kwargs)


def id_vds(vGS=0, vT=0, NMOSFET=True, CLM=0.0, plots_dict=None):

    # init
    NMOSFET = bool(NMOSFET)
    N = 1000
    vDS_max = 10
    k = 2

    # range of vGS
    if not NMOSFET:
        vDS_max = -vDS_max
    vDS = np.linspace(0, vDS_max, N)
    iD, colors = get_MOSFET_current(vGS, vDS, vT, k, N, NMOSFET, CLM)

    # first time in
    if plots_dict is None:
        plots_dict = dict().fromkeys(['output characteristic'])

    # ...
    if all(plot_data is None for plot_data in plots_dict.values()):
        
        lin_x = bq.LinearScale()
        lin_y = bq.LinearScale()

        # plot
        plots_dict['output characteristic'] = bq.FlexLine(
            x=np.zeros(N), y=np.zeros(N),
            color=colors,
            scales={
                "x": lin_x,
                "y": lin_y,
                "color": bq.OrdinalColorScale(scheme='Dark2', domain=[0, 1, 2])
            },
            stroke_width=4,
        )

        ax_x = bplt.Axis(
            label='v_{DS}', scale=lin_x, grid_lines='solid', tick_format='0.2f'
        )
        ax_y = bplt.Axis(
            label='i_D', scale=lin_y, orientation='vertical',
            tick_format='0.2f'
        )
        fig = bplt.figure(
            title="Output characteristic",
            axes=[ax_x, ax_y],
            scales={
                'x': bq.LinearScale(min=0, max=8),
                'y': bq.LinearScale(min=0, max=15),
            },
            marks=list(plots_dict.values()),
        )
        fig.layout.width = '600px'
        # bplt.legend()
        figures = [fig]
    else:
        figures = []

    plots_dict['output characteristic'].x = vDS
    plots_dict['output characteristic'].y = iD
    plots_dict['output characteristic'].color = colors

    return figures, plots_dict


class MOSFETTransferWidgetizer(Widgetizer):
    def __init__(self):
        self.independent_sliders = {
            'vDS': FloatSlider(
                description='vDS', value=0, min=-10, max=10, step=0.1,
                readout_format='.2e', orientation='vertical',
            ),
            'vT': FloatSlider(
                description='vT', value=2, min=-4, max=4, step=0.1,
                readout_format='.2e', orientation='vertical',
            ),
            'NMOSFET': IntSlider(
                description='NMOSFET', value=1, min=0, max=1, step=1,
                readout_format='d', orientation='vertical',
            ),
            'CLM': FloatSlider(
                description='CLM', value=0.0, min=0.0, max=0.02, step=0.002,
                readout_format='.2e', orientation='vertical',
            ),
        }
        super().__init__()

    @staticmethod
    def local_plotter(**kwargs):
        return id_vgs(**kwargs)


def id_vgs(vDS=0, vT=0, NMOSFET=True, CLM=0.0, plots_dict=None):

    # init
    NMOSFET = bool(NMOSFET)
    N = 1000
    vGS_mag_max = 3
    k = 2

    # range of vGS
    vGS = np.linspace(-vGS_mag_max, vGS_mag_max, N)
    iD, colors = get_MOSFET_current(vGS, vDS, vT, k, N, NMOSFET, CLM)

    # you require vDS >= 0 for NMOS and vDS <= 0 for PMOS
    if (vDS < 0) == NMOSFET:
        iD = np.nan*iD  # or nan??

    # first time in
    if plots_dict is None:
        plots_dict = dict().fromkeys(['transfer characteristic', 'zero volts'])

    # ...
    if all(plot_data is None for plot_data in plots_dict.values()):
        
        lin_x = bq.LinearScale()
        lin_y = bq.LinearScale()

        # plot
        plots_dict['zero volts'] = bq.Lines(
            x=np.zeros(N), y=np.linspace(0, max(iD), N), colors=['black'],
            opacities=[0.5],
            line_style='dashed',
            scales={
                "x": lin_x,
                "y": lin_y,
            },
        )
        plots_dict['transfer characteristic'] = bq.FlexLine(
            x=np.zeros(N), y=np.zeros(N),
            color=colors,
            scales={
                "x": lin_x,
                "y": lin_y,
                "color": bq.OrdinalColorScale(scheme='Dark2', domain=[0, 1, 2])
            },
            stroke_width=4,
        )

        ax_x = bplt.Axis(
            label='v_{GS}', scale=lin_x, grid_lines='solid', tick_format='0.2f'
        )
        ax_y = bplt.Axis(
            label='i_D', scale=lin_y, orientation='vertical',
            tick_format='0.2f'
        )
        fig = bplt.figure(
            title="Transfer characteristic",
            axes=[ax_x, ax_y],
            scales={
                'x': bq.LinearScale(min=-5, max=5),
                'y': bq.LinearScale(min=0, max=10),
            },
            marks=list(plots_dict.values()),
        )
        # bplt.legend()
        fig.layout.width = '600px'
        figures = [fig]
    else:
        figures = []

    plots_dict['transfer characteristic'].x = vGS
    plots_dict['transfer characteristic'].y = iD
    plots_dict['transfer characteristic'].color = colors

    plots_dict['zero volts'].y = np.linspace(0, max(iD), N)

    return figures, plots_dict


def get_MOSFET_voltage(vGS, vT, vDS_max, iD_max, k, N, NMOSFET):

    # useful variable
    vDSS = vGS - vT

    # The load line imposes an extra constraint,
    #   iD = iD_max*(1 - vDS/vDS_max),
    # which we can use to fix vDS once vGS has been fixed
    
    # CUT-OFF:
    #   iD = 0 = iD_max*(1 - vDS/vDS_max) => vDS = vDS_max
    vDS_cutoff = vDS_max + 0*vGS
    
    # TRIODE
    #   iD = k*(vDSS*vDS - vDS^2/2) = iD_max*(1 - vDS/vDS_max),
    #   => (k/2)*vDS^2 - (k*vDSS + iD_max/vDS_max)*vDS + iD_max = 0
    #   => vDS = (k*vDSS + iD_max/vDS_max)/k
    #          +/- sqrt{(k*vDSS + iD_max/vDS_max)^2 - 2*k*iD_max}/k
    #          = [vDSS + iD_max/(k*vDS_max)]
    #          +/- sqrt{[vDSS + iD_max/(k*vDS_max)]^2 - 2*iD_max/k}
    b = (vDSS + iD_max/(k*vDS_max))
    d = b**2 - 2*iD_max/k
    # Because we are computing this for *all* vGS, we can get negative
    #  discriminants; zero these to avoid errors:
    d[d < 0] = 0
    if NMOSFET:
        vDS_triode = b - d**(1/2)
    else:
        vDS_triode = b + d**(1/2)
    
    # SATURATION
    #   iD = k/2*vDSS^2 = iD_max*(1 - vDS/vDS_max)
    #   => vDS = (1 - k/2*vDSS**2/iD_max)*vDS_max
    vDS_sat = (1 - k/2*vDSS**2/iD_max)*vDS_max

    # booleans
    CUTOFF = np.array(vDSS < 0)
    SATURATION = np.array(vDS_sat > vDSS)

    if not NMOSFET:
        CUTOFF = ~CUTOFF
        SATURATION = ~SATURATION

    # put all the voltages together into one array
    vDS = CUTOFF*vDS_cutoff + (~CUTOFF)*((~SATURATION)*vDS_triode + SATURATION*vDS_sat)

    # colors for the different regimes
    colors = get_MOSFET_colors(CUTOFF, SATURATION, N)

    return vDS, colors


def get_MOSFET_current(vGS, vDS, vT, k, N, NMOSFET, CLM):

    # useful variables
    vDSS = vGS - vT
    vExcess = vDSS - vDS
    
    # booleans
    CUTOFF = np.array(vDSS < 0)
    SATURATION = np.array(vExcess <= 0)
    
    if not NMOSFET:
        CUTOFF = ~CUTOFF
        SATURATION = ~SATURATION
        CLM = -CLM

    # current in different regimes
    iD_cutoff = 0*vGS
    iD_triode = k*(vDSS*vDS - vDS**2/2)
    iD_sat = k/2*vDSS**2*(1.0 - CLM*vExcess)

    # put all the currents together into one array
    iD = CUTOFF*iD_cutoff + (~CUTOFF)*((~SATURATION)*iD_triode + SATURATION*iD_sat)

    # colors for the different regimes
    colors = get_MOSFET_colors(CUTOFF, SATURATION, N)
    
    return iD, colors


def get_MOSFET_colors(CUTOFF, SATURATION, N):
    # might not need N....
    return (
        CUTOFF*np.ones(N)*1 + ~CUTOFF*(
            ~SATURATION*np.ones(N)*2 + SATURATION*np.ones(N)*0
        )
    )


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
    vDS=1, omega=0.1, phi=0, t=0, plots_dict=None,
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
            x=np.zeros(N), y=np.zeros(N), color=np.linspace(0, 1, N),
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
            x=np.zeros(N), y=np.zeros(N), color=np.linspace(0, 1, N),
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
            x=np.zeros(N), y=np.zeros(N), color=np.linspace(0, 1, N),
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
    plots_dict['counterclockwise'].x = np.real(f1(np.linspace(t0, t, N)))
    plots_dict['counterclockwise'].y = np.imag(f1(np.linspace(t0, t, N)))

    plots_dict['clockwise'].x = np.real(f2(np.linspace(t0, t, N)))
    plots_dict['clockwise'].y = np.imag(f2(np.linspace(t0, t, N)))
    
    plots_dict['average'].x = np.real(f3(np.linspace(t0, t, N)))
    plots_dict['average'].y = np.zeros(N)
    
    return figures, plots_dict


class RLC_circuit():
    @auto_attribute
    def __init__(self, R, L, C, PARALLEL=True):
        
        # ...
        pass
    
    @property
    def tau(self):
        return self.R*self.C if self.PARALLEL else self.L/self.R

    @property
    def sigma(self):
        return 1/(2*self.tau)
    
    @property
    def bandwidth(self):
        return 2*self.sigma

    @property
    def w0(self):
        return 1/(self.L*self.C)**(1/2)
        
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
        s1 = +gg - self.bandwidth/2
        s2 = -gg - self.bandwidth/2
        return [s1, s2]

    @property
    def max_pole_magnitude(self):
        s1, s2 = self.poles
        return max(np.abs(s1), np.abs(s2))

    def temporal_response(self, x0=0, xdot0=0, t_max=0.2, T=10000):
        legend_str = []
        
        t = np.linspace(0, t_max, T)
        
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


class RLCWidgetizer(Widgetizer):

    def __init__(self):

        # initial values (these will be adjusted by sliders)
        R = 50
        L = 1
        C = 100e-6

        # hard-coded to be consistent with the parameter ranges (min/max)
        self.frequencies = np.logspace(-3, 7, 1000)
        self.t_max = 0.2
        self.zeros = [0]

        self.circuit = RLC_circuit(R, L, C, PARALLEL=True)

        self.independent_sliders = {
            'R': FloatSlider(
                description='R', value=self.circuit.R, min=0.5, max=1000,
                step=0.5, readout_format='.2e', orientation='vertical',
            ),
            'L': FloatSlider(
                description='L', value=self.circuit.L, min=1/3, max=10,
                step=1/3, orientation='vertical',
            ),
            'C': FloatSlider(
                description='C', value=self.circuit.C, min=10e-6, max=0.25,
                step=10e-6, readout_format='.2e', orientation='vertical',
            ),
            'x0': FloatSlider(
                description='$x_0$', value=0, min=0, max=10, step=0.1,
                orientation='vertical',
            ),
            'xdot0': FloatSlider(
                description=r'$\dot x_0$', value=1000, min=0, max=10000,
                step=100, orientation='vertical',
            ),
            'PARALLEL': IntSlider(
                description='PARALLEL', value=int(self.circuit.PARALLEL),
                min=0, max=1, step=1, readout_format='d',
                orientation='vertical',
            ),
        }
        
        # these must be properties of self.circuit (see set_slider_values)
        self.dependent_sliders = {
            'w0': FloatSlider(description=r'$\omega_0$', orientation='vertical'),
            'sigma': FloatSlider(description=r'$\sigma$', orientation='vertical'),
            'bandwidth': FloatSlider(description=r'$B_{\omega}$', orientation='vertical'),
            'quality_factor': FloatSlider(description='$Q$', orientation='vertical'),
            'w_L': FloatSlider(description=r'$\omega_L$', orientation='vertical'),
            'w_H': FloatSlider(description=r'$\omega_H$', orientation='vertical'),
            'max_pole_magnitude': FloatSlider(description=r'$\max|p|$', orientation='vertical'),
        }
        self.set_slider_values()
        self.set_slider_extrema()
        
        super().__init__()

    def set_slider_values(self):
        for key, slider in self.dependent_sliders.items():
            slider.value = getattr(self.circuit, key)

    def set_slider_extrema(self):
        '''
        Brute-force calculation of maximima and minima, assuming the min and
         max occur at some extreme values of the independent parameters.
        
        (But is this true??)
        '''

        # shortcut notation
        ind_slider = self.independent_sliders
        for R in [ind_slider['R'].min, ind_slider['R'].max]:
            for L in [ind_slider['L'].min, ind_slider['L'].max]:
                for C in [ind_slider['C'].min, ind_slider['C'].max]:
                    test_circuit = RLC_circuit(R, L, C, PARALLEL=True)
                    for key, dep_slider in self.dependent_sliders.items():
                        circuit_val = getattr(test_circuit, key)
                        dep_slider.min = min(dep_slider.min, circuit_val)
                        dep_slider.max = max(dep_slider.max, circuit_val)


    # @staticmethod
    def local_plotter(self, **kwargs):

        # it seems like you could really just create a new circuit here
        self.circuit.R = kwargs.pop('R')
        self.circuit.L = kwargs.pop('L')
        self.circuit.C = kwargs.pop('C')
        PARALLEL = kwargs.pop('PARALLEL')

        if PARALLEL != self.circuit.PARALLEL:
            # the extrema are different for the other RLC
            self.circuit.PARALLEL = PARALLEL
            self.set_slider_extrema()

        return RLC_plotter(
            self.circuit, self.frequencies, self.t_max, self.zeros, **kwargs
        )


def RLC_plotter(
    circuit, frequencies, t_max, zeros, x0=0, xdot0=0, plots_dict=None
):
    if plots_dict is None:
        plots_dict = dict().fromkeys(
            ['temporal', 'Bode magnitude', 'Bode phase', 'pole-zero']
        )

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

    return figures, plots_dict


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
                description=r'$\dot x_0$', value=1000, min=0, max=10000,
                step=100, orientation='vertical',
            ),
        }
        super().__init__()

    @staticmethod
    def local_plotter(**kwargs):
        return convolution_plotter(**kwargs)


def convolution_plotter(R=1, L=1, C=1, x0=0, xdot0=0, plots_dict=None):
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
