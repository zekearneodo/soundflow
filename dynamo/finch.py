import numpy as np
import matplotlib.pyplot as plt

import dynamo.ode as ode


def takens_finch(v, pars):
    g = pars['gamma']
    gg = g * g
    [x, y, i_1, i_2, i_3] = v

    return np.array([
        y,
        pars['alpha_1'] * gg + pars['beta_1'] * gg * x - gg * x * x * x - g * x * x * y + gg * x * x - g * x * y,
        i_2,
        -pars['Lg_inv'] * pars['Ch_inv'] * i_1 - pars['Rh'] * (pars['Lb_inv'] + pars['Lg_inv']) * i_2 + (
            pars['Lg_inv'] * pars['Ch_inv'] - pars['Rb'] * pars['Rh'] * pars['Lb_inv'] * pars['Lg_inv']) * i_3 +
        pars['Lg_inv'] * pars['dV_ext'] + pars['Rh'] * pars['Lg_inv'] * pars['Lb_inv'] * pars['V_ext'],
        -pars['Lb_inv'] / pars['Lg_inv'] * i_2 - pars['Rb'] * pars['Lb_inv'] * i_3 + pars['Lb_inv'] * pars['V_ext']
    ])


def finch(pars, int_par_stream, s_f=44100., x_0=None):
    """
    :param pars: dictionary with parameters, includyng ['sys'] with parameters of the diff equation
    :param int_par_stream: numpy array stream of parameters for integration with columns [alpha, beta, envelope]
    :param x_0: state vector dim 5 of source + OEC (2+3): [x, y, i_1, i_2, i_3]
    :return: pre_out (air pressure at the output of the beak)
    """

    if x_0 is None:
        x = np.array([5.E-12,
                      1.E-11,
                      1.E-12,
                      1.E-11,
                      1.E-12
                      ])
    else:
        x = x_0

    stream_len = int_par_stream.shape[0]

    steps_per_sample = 20
    d_t = 1. / (s_f * steps_per_sample)

    total_steps = stream_len * steps_per_sample
    pre_out = np.zeros([stream_len, 3])
    t_fractional = 0.

    a = 0
    bf = 1
    bb = 2
    cf = 3
    cb = 4
    db = 5
    df = 6
    pars = compute_tract_pars(pars, d_t)
    tract_buffers = np.zeros([total_steps + pars['max_tau'], 7])

    syrinx = ode.System(takens_finch, x, t_0=pars['max_tau'], field_pars=pars['sys'], dt=d_t, sampling_rate=s_f)

    t_fractional = 0.
    sampling_t = 0
    t_samples = 0

    for step in np.arange(pars['max_tau'], pars['max_tau'] + total_steps):

        if sampling_t == 0:
            pars['sys']['alpha_1'] = int_par_stream[t_samples, 0]
            beta_1 = int_par_stream[t_samples, 1]
            pars['sys']['envelope'] = int_par_stream[t_samples, 2]
            pre_out[t_samples, 0] = x[4]
            pre_out[t_samples, 1] = x[0]
            pre_out[t_samples, 2] = pars['sys']['beta_1']
            t_samples += 1
            t_fractional += d_t * steps_per_sample
            sampling_t = steps_per_sample

        pars['sys']['noise'] = np.random.normal()
        pars['A_1'] = np.sqrt(pars['sys']['envelope']) * (1. + 0.1*pars['sys']['noise'])
        pars['sys']['beta_1'] = beta_1 * (1. + 0.1 * pars['sys']['noise'])


        db_old = tract_buffers[step, db]
        tract_buffers[step, a] = 0.5 * pars['A_1'] * x[1] + \
                                 tract_buffers[step - pars['tau_1'], bb]

        tract_buffers[step, bb] = pars['r_12'] * tract_buffers[step - pars['tau_1'], a] + \
                                  pars['t_21'] * tract_buffers[step - pars['tau_2'], cb]

        tract_buffers[step, bf] = pars['t_12'] * tract_buffers[step - pars['tau_1'], a] + \
                                  pars['r_21'] * tract_buffers[step - pars['tau_2'], cb]

        tract_buffers[step, cb] = pars['r_23'] * tract_buffers[step - pars['tau_2'], bf] + \
                                  pars['t_32'] * tract_buffers[step - pars['tau_3'], db]

        tract_buffers[step, cf] = pars['t_23'] * tract_buffers[step - pars['tau_2'], bf] + \
                                  pars['r_32'] * tract_buffers[step - pars['tau_3'], db]

        tract_buffers[step, db] = -pars['r_out'] * tract_buffers[step - pars['tau_3'], cf]

        ddb = (tract_buffers[step, db] - db_old) / d_t

        pars['sys']['V_ext'] = tract_buffers[step, a]
        pars['sys']['dV_ext'] = ddb

        syrinx.update_pars(pars['sys'])
        x = syrinx.next()
        sampling_t -= 1

    return pre_out


def compute_tract_pars(pars, d_t, v_sound=35000):
    for t in ['12', '21', '23', '32']:
        pars['r_' + t] = (pars['S_' + t[0]] - pars['S_' + t[1]]) / (pars['S_' + t[0]] + pars['S_' + t[1]])
        pars['t_' + t] = 1. + pars['r_' + t]
        pars['tau_' + t[0]] = int(pars['l_' + t[0]] / (v_sound * d_t))

    pars['max_tau'] = np.max(np.array([pars['tau_' + i] for i in ['1', '2', '3']]))

    return pars


def main():
    sys_pars = {'alpha_1': 0.15,
                'beta_1': 0.15,
                'alpha_2': 0.15,
                'beta_2': 0.15,
                'gamma': 23500.,
                'Ch_inv': 4.5E-10,
                'Lb_inv': 1.1E-4,
                'Lg_inv': 0.012195,
                'Rb': 5.0E8,
                'Rh': 6.0E5,
                'V_ext': 0.,
                'dV_ext': 0.,
                'noise': 0.,
                'envelope': 0.
                }

    vocal_pars = {'sys': sys_pars,
                  'S_1': 0.2,
                  'S_2': 0.2,
                  'S_3': 0.2,
                  'l_1': 1.5,
                  'l_2': 1.5,
                  'l_3': 1.0,
                  'r_out': 0.1,
                  'r_12': None,
                  'r_21': None,
                  'r_23': None,
                  'r_32': None,
                  't_12': None,
                  't_21': None,
                  't_23': None,
                  't_32': None,
                  'tau_1': None,
                  'tau_2': None,
                  'tau_3': None,
                  'max_tau': None,
                  'A_1': 0.,
                  'A_2': None,
                  'A_3': None}

    x = np.array([5.E-12,
                  1.E-11,
                  1.E-12,
                  1.E-11,
                  1.E-12
                  ])

    # make a test run of a segment
    segment_samples = 4410  # 100ms
    alpha_values = -0.15 * np.ones(segment_samples)
    beta_values = -0.15 * np.ones(segment_samples)
    env_values = 0.5 * np.ones(segment_samples)

    par_stream = np.array([alpha_values, beta_values, env_values]).T

    song_synth = finch(vocal_pars, par_stream, s_f=44100., x_0=x)
    plt.plot(song_synth)


if __name__ == '__main__':
    main()