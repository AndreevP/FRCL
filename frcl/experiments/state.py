import warnings

class LevelState:

    def __init__(self, **kwargs):

        for k, v in kwargs.items():
            setattr(self, k, v)


class State(object):

    def __init__(self):
        super().__setattr__('level_states', [])
    
    def attach_current(self, **kwargs):
        if len(self.level_states) == 0:
            self.attach_level(**kwargs)
            return
        for key, value in kwargs.items():
            setattr(self, key, value)

    
    def attach_level(self, **kwargs):
        self.level_states.append(LevelState())
        self.attach_current(**kwargs)
    
    def detach_level(self):
        if len(self.level_states) > 0:
            del self.level_states[-1]
            return
        raise Exception('State is already empty!')

    def get_level_dict(self):
        if len(self.level_states) > 0:
            return self.level_states[-1].__dict__
        raise Exception('State is already empty!')

    def detach_current(self, *args):
        if len(self.level_states) == 0:
            return
        for arg in args:
            if hasattr(self.level_states[-1], arg):
                self.level_states[-1].__delattr__(arg)

    
    def __setattr__(self, name, value):
        #print('setattr call, name: {}, value: {}'.format(name, value))
        for i, level_st in enumerate(self.level_states):
            if hasattr(level_st, name):
                #level_value = getattr(level_st, name)
                if i < len(self.level_states) - 1:
                    warnings.warn("Hide definition of name '{}' on level {}".format(name, i))
                else:
                    warnings.warn("Change definintion of name '{}'".format(name))
        if len(self.level_states) == 0:
            self.attach_level()
        self.level_states[-1].__setattr__(name, value)
    
    def __getattr__(self, name):
        #print('getattr call, {}'.format(name))
        if name == '__iter__':
            raise Exception()
        for i in range(len(self.level_states)):
            n_level = len(self.level_states) - 1 - i
            level_st = self.level_states[n_level]
            if hasattr(level_st, name):
                return getattr(level_st, name)
        raise AttributeError("Name '{}' doesn't presented in the state".format(name))