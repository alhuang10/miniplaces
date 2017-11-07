from FoxNet import *

fox = FoxNet()

fox.load_state_dict(torch.load("current_best_model_weights"))
fox.cuda()

evaluate_foxnet(fox, True)
