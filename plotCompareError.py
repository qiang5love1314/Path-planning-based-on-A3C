import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt

def main():
    figure, ax = plt.subplots()

    '----------Meeting Room-----------'
    # stepLength = ['50', '60', '80', '90', '100']
    # QLerror = [np.nan, 3.033412053886134, 3.100257758124791, 3.3968204467023737, 3.040236773233606]
    # QLstd = [np.nan, 1.4282896454635206, 1.4725494331896125, 1.4968002047249624, 1.59704738898947]
    # GAerror = [3.196366553037086,2.9114208667594648,2.978719808367925,3.19769936112328,3.002680540036201]
    # GAstd = [1.8130749732500901,1.6351234010305054,1.698713720212018,1.848545048375011,1.6774711843986805]
    # GreedyError = [3.028757259496588, 3.0777150180367796, 3.1026195828067014, 3.2262397045032274, 3.052888699580351]
    # GreedyStd = [1.2802458603903843, 1.6541070907746165, 1.3286653921828366, 1.4788432537246576, 1.6263672979910129]
    # A3CError = [2.923807508915902, 2.8773282832081355, 2.950146870797033, 2.8558417467909347, 2.948270507215346]
    # A3CStd = [1.578020801766246, 1.4024279563930357, 1.3744938852997755, 1.4147785949899223, 1.5687259213400417]
    # DDPGError = [3.0062666985303226, 2.9041050204722496, 2.9605341650906074, 2.873424897096752, 2.9473102260641078]
    # DDPGStd = [1.6391340815526951, 1.2756857097529455, 1.6565136453803389, 1.6431157478231782, 1.5484709979001672]

    # '----------Lab-----------'
    stepLength = [ '130', '150', '180', '200']
    QLerror = [ 4.2542812502920994, 4.388643991207664, 4.195432277500014, 4.488288549696721]
    QLstd = [ 2.2733160896393394, 2.2008785560400352, 2.343889236911813, 2.262386106450731]
    GAerror = [ 4.254281250292099, 4.388643991207664, 4.195432277500014, 4.488288549696721]
    GAstd = [ 2.2733160896393394, 2.2008785560400352, 2.343889236911813, 2.262386106450731]
    GreedyError = [ 4.539516936414378, 4.267152331217189, 4.279420972730923, 4.238804340916964]
    GreedyStd = [ 2.188957739200788, 2.003591707955418, 2.121208355666816, 2.154582386316082]
    A3CError = [ 4.418406232225777, 4.224087114051174, 4.218495687383834, 4.219501631998242]
    A3CStd = [ 2.2139540684098256, 2.1298607473041105, 2.1357625419329733, 2.1592518328264]
    DDPGError = [4.355805553456392, 4.574155992111455, 4.335134027572516, 4.356651485533076]
    DDPGStd = [2.1827638398320737, 2.2542120485506376, 2.2390470546602845, 2.029060825505839]

    plt.errorbar(stepLength, QLerror, QLstd, fmt='-o', ecolor='c', color='c', elinewidth=2, capsize=4, label='Q-Learning')
    plt.errorbar(stepLength, DDPGError, DDPGStd, fmt='-*', ecolor='orange', color='orange', elinewidth=2, capsize=4, label='PRM-RL')
    plt.errorbar(stepLength, GAerror, GAstd, fmt='-v', ecolor='b', color='b', elinewidth=2, capsize=4, label='GA')
    plt.errorbar(stepLength, GreedyError, GreedyStd, fmt='-x', ecolor='green', color='green', elinewidth=2, capsize=4, label='Greedy')
    plt.errorbar(stepLength, A3CError, A3CStd, fmt='-p', ecolor='r', color='r', elinewidth=2, capsize=4, label='A3C')
    # fmt:'o' ',' '.' 'x' '+' 'v' '^' '<' '>' 's' 'd' 'p'


    font2 = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 15,
             }
    plt.xlabel('Exploration Step Length', font2)
    plt.ylabel('Localization Error (m)', font2)
    plt.grid(color="grey", linestyle=':', linewidth=0.5)
    plt.tick_params(labelsize=15)
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]
    plt.legend(prop={'family': 'Times New Roman', 'size': 12}, loc='upper right')
    # plt.ylim(0, 7)
    plt.savefig('LabCompareError.pdf', bbox_inches='tight', dpi=500)
    plt.show()

if __name__ == '__main__':
    main()