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
        args.bnds = ((.000, .99),(.000, .99))
        args.params_name = [ 'lr_q', 'tau']
    elif args.agent_name == 'Pi_model_2':
        args.agent = Pi_model_2
        args.bnds = ((.00, .99), (.00, .99), (.00, .7), (0, 15), 
                        (0, 15),    (0, 10),   (0, 10), (0, 10))
        args.params_name = [ 'α_v', 'α_θ', 'α_a', 'β2','β3','β4','β5','β6']

    # if init, do not 
    if len(args.init) > 0:
        args.fit_num = 1 
    return args
