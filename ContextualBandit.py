import vowpalwabbit
import random
import itertools
from tqdm import tqdm

class ContextualBandit:
    """
    Parameters
    ----------
    method   :(str) a sequence of commands to create a contextual bandit object (refer VowpalWabbit documentation)
                https://vowpalwabbit.org/docs/vowpal_wabbit/python/latest/tutorials/python_cats.html 
    
    Attributes
    ----------

    """
    
    def __init__(self,method):
        self.vw = vowpalwabbit.Workspace(method)
        
    def finish(self):
        '''
        To terminate a vowpal wabbit object
        '''
        self.vw = self.vw.finish()
    
    def to_vw_example_format(self,context,actions,cb_label=None):
        '''
        To convert the context, actions required for learning the reward model paramters into vowpal 
        wabbit supported input format
        '''
        if cb_label is not None:
            chosen_action, cost, prob = cb_label
        example_string = ""
        example_string += "shared |Songs "
        for i,con in enumerate(context):
            example_string+=" song{}=".format(i+1)+con
        example_string += "\n"
        for action in actions:
            if cb_label is not None and action == chosen_action:
                example_string += "0:{}:{} ".format(cost, prob)
            example_string += "|Action song={} \n".format(action)
        # Strip the last newline
        return example_string[:-1]
    
    def sample_custom_pmf(self,pmf):
        '''
         To randomly select an action using pmf with support as the action space
        '''
        total = sum(pmf)
        scale = 1 / total
        pmf = [x * scale for x in pmf]
        draw = random.random()
        sum_prob = 0.0
        for index, prob in enumerate(pmf):
            sum_prob += prob
            if sum_prob > draw:
                return index, prob
        
    def get_action(self,context,actions):
        '''
        Generate action from the Vowpall Wabbit contextual bandit object and return the chosen action
        '''
        vw_text_example = self.to_vw_example_format(context,actions)
        #print(vw_text_example)
        pmf = self.vw.predict(vw_text_example)
        chosen_action_index, prob = self.sample_custom_pmf(pmf)
        return actions[chosen_action_index], prob
    