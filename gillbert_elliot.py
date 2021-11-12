# -*- coding: utf-8 -*-
"""
Created on Thu Nov 11 21:46:19 2021

@author: Navya
"""

from sim2net.packet_loss._packet_loss import PacketLoss
from sim2net.utility.validation import check_argument_type


__docformat__ = 'reStructuredText'


#pylint: disable=C0103
class GilbertElliott(PacketLoss):
    """
    This class implements the Gilbert-Elliott packet loss model.
    """

    #: Default value of the `p` parameter.
    __DEFAULT_P = 0.00001333

    #: Default value of the `r` parameter.
    __DEFAULT_R = 0.00601795

    #: Default value of the `h` parameter.
    __DEFAULT_H = 0.55494900

    #: Default value of the `k` parameter.
    __DEFAULT_K = 0.99999900


    def __init__(self, prhk=None):
        """
        *Parameters*:
            - **prhk** (`tuple`): a `tuple` that contains four model
              parameters: :math:`0\\leqslant p,r,h,k\\leqslant 1`, respectively
              (each of type `float`).  The parameters default to the following
              values:
              * :math:`p=0.00001333`,
              * :math:`r=0.00601795`,
              * :math:`h=0.55494900`,
              * :math:`k=0.99999900`;
              (which leads to error rate equal to :math:`0.098\\%` and the mean
              packet loss rate equal to :math:`0.1\\%` ([HH08]_)).
        *Raises*:
            - **ValueError**: raised when the given value any model parameter
              is less than zero or greater that one.
        (At the beginning the model is in the ``G`` state.)
        """
        super(GilbertElliott, self).__init__(PacketLoss.__name__)
        if prhk is None:
            p = float(GilbertElliott.__DEFAULT_P)
            r = float(GilbertElliott.__DEFAULT_R)
            b = float(1.0 - GilbertElliott.__DEFAULT_H)
            g = float(1.0 - GilbertElliott.__DEFAULT_K)
        else:
            for param in range(4):
                check_argument_type(GilbertElliott.__name__,
                                    'prhk[' + str(param) + ']', float,
                                    prhk[param], self.logger)
                if prhk[param] < 0.0 or prhk[param] > 1.0:
                    raise ValueError('Parameter "prhk[%d]": a value of the' \
                                     ' model parameter cannot be less than' \
                                     ' zero and greater than one but %f given!'
                                     % (param, float(prhk[param])))
            p = float(prhk[0])
            r = float(prhk[1])
            b = float(1.0 - prhk[2])
            g = float(1.0 - prhk[3])
        # ( current state: 'G' or 'B',
        #   transition probability,
        #   current packet error rate )
        self.__state_g = ('G', p, g)
        self.__state_b = ('B', r, b)
        self.__current_state = self.__state_g

    def packet_loss(self):
        """
        Returns information about whether a transmitted packet has been lost or
        can be successfully received by destination node(s) according to the
        Gilbert-Elliott packet loss model.
        *Returns*:
            (`bool`) `True` if the packet has been lost, or `False` otherwise.
        """
        transition = self.random_generator.uniform(0.0, 1.0)
        if transition <= self.__current_state[1]:
            if self.__current_state[0] == 'G':
                self.__current_state = self.__state_b
            else:
                self.__current_state = self.__state_g
        loss = self.random_generator.uniform(0.0, 1.0)
        if loss <= self.__current_state[2]:
            return True
        return False