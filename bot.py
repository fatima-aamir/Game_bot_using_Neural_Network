import joblib
from command import Command
import numpy as np
from buttons import Buttons
import tensorflow as tf
from sklearn.preprocessing import StandardScaler


class Bot:

    def __init__(self):
        #< - v + < - v - v + > - > + Y
        self.fire_code=["<","!<","v+<","!v+!<","v","!v","v+>","!v+!>",">+Y","!>+!Y"]
        self.bot_model = tf.keras.models.load_model('ModelFile.h5')
        self.scaler = joblib.load('ScaledValues.save')
        self.exe_code = 0
        self.start_fire=True
        self.remaining_code=[]
        self.my_command = Command()
        self.buttn= Buttons()

    def fight(self, current_game_state, player):
        if player == "1":
            single_input = np.array([[current_game_state.timer, current_game_state.player2.health, current_game_state.player2.x_coord, current_game_state.player2.y_coord, current_game_state.player1.x_coord,	current_game_state.player1.y_coord,	current_game_state.player1.is_jumping,	current_game_state.player1.is_crouching, abs(current_game_state.player1.x_coord-current_game_state.player1.x_coord), abs(current_game_state.player1.y_coord-current_game_state.player1.y_coord)]])
            scaled_input = self.scaler.transform(single_input)
            predictions = self.bot_model.predict(scaled_input)
            predicted_buttons = (predictions > 0.1)
            predicted_buttons = predicted_buttons.astype(bool)
                
            self.buttn.up=self.buttn.down=self.buttn.left=self.buttn.right=self.buttn.L=self.buttn.R=self.buttn.A=self.buttn.B=self.buttn.X=self.buttn.Y = False

            buttons = ["up", "down", "left", "right", "L", "R", "A", "B", "X", "Y"]

            for i, button in enumerate(buttons):
                if predicted_buttons[0][i]:
                    setattr(self.buttn, button, True)

            self.my_command.player_buttons = self.buttn

        elif player == "2":
            self.my_command.player2_buttons = self.buttn

    
        #Same as original File
        elif player=="2":

            if( self.exe_code!=0  ):
               self.run_command([],current_game_state.player2)
            diff=current_game_state.player1.x_coord - current_game_state.player2.x_coord
            if (  diff > 60 ) :
                toss=np.random.randint(3)
                if (toss==0):
                    #self.run_command([">+^+Y",">+^+Y",">+^+Y","!>+!^+!Y"],current_game_state.player2)
                    self.run_command([">","-","!>","v+>","-","!v+!>","v","-","!v","v+<","-","!v+!<","<+Y","-","!<+!Y"],current_game_state.player2)
                elif ( toss==1 ):
                    self.run_command([">+^+B",">+^+B","!>+!^+!B"],current_game_state.player2)
                else:
                    self.run_command(["<","-","!<","v+<","-","!v+!<","v","-","!v","v+>","-","!v+!>",">+Y","-","!>+!Y"],current_game_state.player2)
            elif ( diff < -60 ) :
                toss=np.random.randint(3)
                if (toss==0):
                    #self.run_command(["<+^+Y","<+^+Y","<+^+Y","!<+!^+!Y"],current_game_state.player2)
                    self.run_command(["<","-","!<","v+<","-","!v+!<","v","-","!v","v+>","-","!v+!>",">+Y","-","!>+!Y"],current_game_state.player2)
                elif ( toss==1):
                    self.run_command(["<+^+B","<+^+B","!<+!^+!B"],current_game_state.player2)
                else:
                    self.run_command([">","-","!>","v+>","-","!v+!>","v","-","!v","v+<","-","!v+!<","<+Y","-","!<+!Y"],current_game_state.player2)
            else:
                toss=np.random.randint(2)  # anyFightActionIsTrue(current_game_state.player2.player_buttons)
                if ( toss>=1 ):
                    if (diff<0):
                        self.run_command(["<","<","!<"],current_game_state.player2)
                    else:
                        self.run_command([">",">","!>"],current_game_state.player2)
                else:
                    self.run_command(["v+R","v+R","v+R","!v+!R"],current_game_state.player2)
            self.my_command.player2_buttons=self.buttn


        return self.my_command
