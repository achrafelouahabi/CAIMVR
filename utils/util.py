import numpy as np

def normalize(x):
    """ Normalize """
    x = (x - np.min(x)) / (np.max(x) - np.min(x))
    return x

def cal_std(*arg, logger=None):
    """ print clustering results """
    # Fonction pour afficher avec logger ou print
    def log(message):
        if logger is not None:
            logger.info(message)
        else:
            print(message)
    
    if len(arg) == 3:
        log(arg[0])
        log(arg[1])
        log(arg[2])
        output = """                      ACC {:.2f} std {:.2f}
                     NMI {:.2f} std {:.2f}
                      ARI {:.2f} std {:.2f}""".format(np.mean(arg[0]) * 100, np.std(arg[0]) * 100, 
                                                     np.mean(arg[1]) * 100, np.std(arg[1]) * 100, 
                                                     np.mean(arg[2]) * 100, np.std(arg[2]) * 100)
        log(output)
        output2 = str(round(np.mean(arg[0]) * 100, 2)) + ',' + str(round(np.std(arg[0]) * 100, 2)) + ';' + \
                  str(round(np.mean(arg[1]) * 100, 2)) + ',' + str(round(np.std(arg[1]) * 100, 2)) + ';' + \
                  str(round(np.mean(arg[2]) * 100, 2)) + ',' + str(round(np.std(arg[2]) * 100, 2)) + ';'
        log(output2)
        return round(np.mean(arg[0]) * 100, 2), round(np.mean(arg[1]) * 100, 2), round(np.mean(arg[2]) * 100, 2)
    
    elif len(arg) == 1:
        log(arg)
        output = """ACC {:.2f} std {:.2f}""".format(np.mean(arg) * 100, np.std(arg) * 100)
        log(output)

def cal_HAR(*arg, logger=None):
    """ print classification results for HAR """
    # Fonction pour afficher avec logger ou print
    def log(message):
        if logger is not None:
            logger.info(message)
        else:
            print(message)
    
    if len(arg) == 5:
        log(arg[0])
        log(arg[1])
        log(arg[2])
        log(arg[3])
        log(arg[4])
        output = """                      RGB {:.2f} std {:.2f}
                     Depth {:.2f} std {:.2f}
                      RGB+D {:.2f} std {:.2f}
                     onlyrgb {:.2f} std {:.2f}
                     onlydepth {:.2f} std {:.2f}""".format(np.mean(arg[0]) * 100, np.std(arg[0]) * 100,
                                                           np.mean(arg[1]) * 100, np.std(arg[1]) * 100,
                                                           np.mean(arg[2]) * 100, np.std(arg[2]) * 100,
                                                           np.mean(arg[3]) * 100, np.std(arg[3]) * 100,
                                                           np.mean(arg[4]) * 100, np.std(arg[4]) * 100)
        log(output)
    return

def cal_classify(*arg, logger=None):
    """ print classification results """
    # Fonction pour afficher avec logger ou print
    def log(message):
        if logger is not None:
            logger.info(message)
        else:
            print(message)
    
    if len(arg) == 4:
        log(arg[0])
        log(arg[1])
        log(arg[2])
        log(arg[3])
        output = """                      ACC {:.2f} std {:.2f}
                     Precision {:.2f} std {:.2f}
                      F-measure {:.2f} std {:.2f}
                      AUC {:.2f} std {:.2f}
                      """.format(np.mean(arg[0]) * 100, np.std(arg[0]) * 100,
                                                           np.mean(arg[1]) * 100, np.std(arg[1]) * 100, 
                                                           np.mean(arg[2]) * 100, np.std(arg[2]) * 100, 
                                                           np.mean(arg[3]) * 100, np.std(arg[3]) * 100)
        log(output)

        return round(np.mean(arg[0]) * 100, 2), round(np.mean(arg[1]) * 100, 2), round(np.mean(arg[2]) * 100, 2), round(np.mean(arg[3]) * 100, 2)
    elif len(arg) == 1:
        log(arg)
        output = """ACC {:.2f} std {:.2f}""".format(np.mean(arg) * 100, np.std(arg) * 100)
        log(output)
    return