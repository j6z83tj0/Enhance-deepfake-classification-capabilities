from .base_options import BaseOptions


class TestOptions(BaseOptions):
    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        parser.add_argument('--model_path')
        parser.add_argument('--no_resize', action='store_true')
        parser.add_argument('--no_crop', action='store_true')
        parser.add_argument('--eval', action='store_true', help='use eval mode during test time.')
        parser.add_argument('--eval_mode', action='store_true', help="In evalmode, preprocessing is applied to all images before, otherwise, only a random 10% of them will undergo preprocessing.")
        parser.add_argument('--threshold',type=float,default=0.5,help='adjust testing threshold')
        
        self.isTrain = False
        return parser
