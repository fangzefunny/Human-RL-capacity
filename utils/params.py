from utils.agents import * 


def set_hyperparams(args):
    '''Set up hyperparams

    This function pass hyper-parameter from
    parser. Where we need to choose.
    '''
    args.eps = 1e-20
    args.init = []
    if args.agent_name == 'optimal':
        args.agent = RLbaseline
        args.bnds = ((.000, .99),(.000, .99))
        args.params_name = [ 'lr_q', 'tau']
    if args.agent_name == 'RLbaseline':
        args.agent = RLbaseline
        args.bnds = ((.000, .99),(.000, .99), (.00,.1))
        args.params_name = [ 'lr_q', 'tau','α_a']
    elif args.agent_name == 'Pi_model_1':
        args.agent = Pi_model_1
        args.bnds = ((.00, .99), (.00, .99), (.00, .7), (0, 10))
        args.params_name = [ 'α_v', 'α_θ', 'α_a', 'β']
    elif args.agent_name == 'Pi_model_2':
        args.agent = Pi_model_2
        args.bnds = ((.00, .99), (.00, .99), (.00, .7), (0, 15), 
                        (0, 15),    (0, 10),   (0, 10), (0, 10))
        args.params_name = [ 'α_v', 'α_θ', 'α_a', 'β2','β3','β4','β5','β6']
    elif args.agent_name == 'Psi_Pi_model_3':
        args.agent = Psi_Pi_model_3
        args.bnds = ((.00, .99),  (.00, 1.), (.00, .7), (0, .15), (0, 55),
                        (0, 45),    (0, 25),   (0, 16), (0, 15))
        args.params_name = [ 'α_q', 'α_π', 'α_a', 'ε','β2','β3','β4','β5','β6']
        args.init = ( .154, .95, .015, .042, 46, 34, 19, 11, 9)
    elif args.agent_name == 'Pi_Rep_Grad':
        args.agent = Pi_Rep_Grad
        args.bnds = ((.00, .99),  (.00, .7), (.00, .7), (0, .15), (0, 15),
                        (0, 45),    (0, 25),   (0, 16), (0, 15) )
        args.params_name = [ 'α_v', 'α_q', 'α_a', 'ε',
                             'β2','β3','β4','β5','β6']
        args.init = ( 0.0,0.03752330382945414,0.015463893864585403, .01,
                      5.959777727231302,5.632271593952316,5.204268368974834,4.809842476104097,4.481592093042523)

    # if init, do not 
    if len(args.init) > 0:
        args.fit_num = 1 
    return args
