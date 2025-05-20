import random

# Template sampler
class Template:
    def __init__(self, templates):
        self.templates = templates
        self.idx = 0
        
    def _reset(self):
        random.shuffle(self.templates)
        self.idx = 0

    def get_template(self):
        if self.idx == len(self.templates):
            self._reset()

        template = self.templates[self.idx]
        self.idx += 1
        return template
    
