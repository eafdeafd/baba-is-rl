from baba.envs import MakeWinEnv, TwoRoomEnv, TwoRoomMakeYouEnv, TwoRoomMakeWallWinEnv

# yeah, these were super tedious and chatgpt'd
def init_goto_win_envs(width, height):
    """ Initialize GotoWin environments with different configurations. """
    return {
        "easy": [MakeWinEnv(width=width, height=height, break_win_rule=False, distractor_obj=False, distractor_rule_block=False)],
        "medium": [
            MakeWinEnv(width=width, height=height, break_win_rule=False, distractor_obj=True, distractor_rule_block=False),
            MakeWinEnv(width=width, height=height, break_win_rule=False, distractor_obj=False, distractor_rule_block=True)
        ],
        "hard": [
            MakeWinEnv(width=width, height=height, break_win_rule=False, distractor_obj=True, distractor_rule_block=True, irrelevant_rule_distractor=True),
            MakeWinEnv(width=width, height=height, break_win_rule=False, distractor_obj=True, distractor_win_rule=True)
        ]
    }

def init_make_win(width, height):
    """ Initialize different MakeWin environments. """
    return {
        "easy": [
            MakeWinEnv(width=width, height=height, distractor_obj=False, distractor_rule_block=False)
        ],
        "medium": [
            MakeWinEnv(width=width, height=height, distractor_rule_block=False),
            MakeWinEnv(width=width, height=height, distractor_obj=False)
        ],
        "hard": [
            MakeWinEnv(width=width, height=height, distractor_rule_block=True, irrelevant_rule_distractor=True)
        ]
    }

def init_two_room(width, height):
    """ Initialize TwoRoom environments related to the 'break_stop' theme with varying difficulties. """
    base_config = {'width': width, 'height': height, 'break_stop_rule': False, 'obj1_pos': "right_anywhere", 'obj2_pos': "right_anywhere"}
    return {
        "easy": [
            TwoRoomEnv(**base_config, distractor_obj=False, distractor_rule_block=False)
        ],
        "medium": [
            TwoRoomEnv(**base_config, distractor_rule_block=False),
            TwoRoomEnv(**base_config, distractor_obj=False)
        ],
        "hard": [
            TwoRoomEnv(**base_config, distractor_obj=True, distractor_rule_block=True, irrelevant_rule_distractor=True)
        ]
    }


def init_two_room_envs_break_stop(width, height):
    """ Initialize TwoRoom environments with various configurations. """
    base_config = {'width': width, 'height': height, 'obj1_pos': "left_anywhere", 'obj2_pos': "left_anywhere", 'break_stop_rule': True}
    return {
        "easy": [TwoRoomEnv(**base_config, distractor_obj=False, distractor_rule_block=False)],
        "medium": [
            TwoRoomEnv(**base_config, distractor_obj=True, distractor_rule_block=False),
            TwoRoomEnv(**base_config, distractor_obj=False, distractor_rule_block=True)
        ],
        "hard": [
            TwoRoomEnv(**base_config, distractor_obj=True, distractor_rule_block=True, irrelevant_rule_distractor=True),
            #TwoRoomEnv(**base_config, distractor_obj=True, distractor_win_rule=True)
        ]
    }

def init_two_room_envs_break_stop_make_win(width, height):
    """ Initialize TwoRoom environments with various configurations. """
    base_config = {'width': width, 'height': height, 'obj1_pos': "left_anywhere", 'obj2_pos': "left_anywhere", 'break_stop_rule': True, 'break_win_rule': True}
    return {
        "easy": [TwoRoomEnv(**base_config, distractor_obj=False, distractor_rule_block=False)],
        "medium": [
            TwoRoomEnv(**base_config, distractor_obj=True, distractor_rule_block=False),
            TwoRoomEnv(**base_config, distractor_obj=False, distractor_rule_block=True)
        ],
        "hard": [
            TwoRoomEnv(**base_config, distractor_obj=True, distractor_rule_block=True, irrelevant_rule_distractor=True),
            #TwoRoomEnv(**base_config, distractor_obj=True, distractor_win_rule=True)
        ]
    }



def init_two_room_anywhere(width, height):
    """ Initialize TwoRoom environments with the 'anywhere' configuration. """
    base_config = {'width': width, 'height': height, 'break_stop_rule': False, 'obj1_pos': "anywhere", 'obj2_pos': "anywhere"}
    return {
        "easy": [
            TwoRoomEnv(**base_config, distractor_obj=False, distractor_rule_block=False)
        ],
        "medium": [
            TwoRoomEnv(**base_config, distractor_obj=True, distractor_rule_block=False),
            TwoRoomEnv(**base_config, distractor_obj=False, distractor_rule_block=True)
        ],
        "hard": [
            TwoRoomEnv(**base_config, distractor_obj=True, distractor_rule_block=True, irrelevant_rule_distractor=True)
        ]
    }

def init_two_room_make_you_make_win(width, height):
    """ Initialize TwoRoom environments that are more challenging. """
    return {
        "easy":[TwoRoomMakeYouEnv(width=width, height=height)],
        "hard": [
            TwoRoomMakeYouEnv(width=width, height=height, break_win_rule=True),
            TwoRoomMakeWallWinEnv(width=width, height=height)
        ]
    }

# # GotoWin Environment with distractor object rule
# def init_goto_win():
#     goto_win = MakeWinEnv(break_win_rule=False, distractor_obj=False, distractor_rule_block=False)
#     goto_win_distr_obj = MakeWinEnv(break_win_rule=False, distractor_obj=True, distractor_rule_block=False)
#     goto_win_distr_rule = MakeWinEnv(break_win_rule=False, distractor_obj=False, distractor_rule_block=True)
#     goto_win_distr_obj_irrelevant_rule = MakeWinEnv(break_win_rule=False, distractor_obj=True, distractor_rule_block=True, irrelevant_rule_distractor=True)
#     goto_win_distr_win_rule = MakeWinEnv(break_win_rule=False, distractor_obj=True, distractor_win_rule=True)
#     return {"easy":[goto_win], "medium":[goto_win_distr_obj, goto_win_distr_rule], "hard":[goto_win_distr_obj_irrelevant_rule, goto_win_distr_win_rule]}

# # Make win Envs
# def init_make_win():
#     make_win = MakeWinEnv()
#     make_win_no_distractor_rule = MakeWinEnv(distractor_rule_block=False)
#     make_win_no_distractor_obj = MakeWinEnv(distractor_obj=False)
#     make_win_no_distractor = MakeWinEnv(distractor_obj=False, distractor_rule_block=False)
#     make_win_irrelevant_distractor_rule = MakeWinEnv(distractor_rule_block=True, irrelevant_rule_distractor=True)

# # TwoRoom GotoWin Environment without any distractors
# def init_two_room():
#     two_room_goto_win = TwoRoomEnv(obj1_pos="left_anywhere", obj2_pos="left_anywhere", break_stop_rule=True, distractor_obj=False, distractor_rule_block=False)
#     two_room_goto_win_distr_obj_rule = TwoRoomEnv(obj1_pos="left_anywhere", obj2_pos="left_anywhere", break_stop_rule=True)
#     two_room_goto_win_distr_rule = TwoRoomEnv(obj1_pos="left_anywhere", obj2_pos="left_anywhere", break_stop_rule=True, distractor_obj=False)
#     two_room_goto_win_distr_obj = TwoRoomEnv(obj1_pos="left_anywhere", obj2_pos="left_anywhere", break_stop_rule=True, distractor_obj=True, distractor_rule_block=False)
#     two_room_goto_win_distr_obj_irrelevant_rule = TwoRoomEnv(obj1_pos="left_anywhere", obj2_pos="left_anywhere", break_stop_rule=True, distractor_obj=True, distractor_rule_block=True, irrelevant_rule_distractor=True)
#     two_room_goto_win_distr_win_rule = TwoRoomEnv(obj1_pos="left_anywhere", obj2_pos="left_anywhere", break_stop_rule=True, distractor_obj=True, distractor_win_rule=True)

# # Variants of TwoRoom BreakStop and GotoWin with various configurations

# two_room_break_stop_goto_win_distr_obj_rule = TwoRoomEnv(break_stop_rule=False, obj1_pos="right_anywhere", obj2_pos="right_anywhere")
# two_room_break_stop_goto_win_distr_obj = TwoRoomEnv(break_stop_rule=False, obj1_pos="right_anywhere", obj2_pos="right_anywhere", distractor_rule_block=False)
# two_room_break_stop_goto_win_distr_rule = TwoRoomEnv(break_stop_rule=False, obj1_pos="right_anywhere", obj2_pos="right_anywhere", distractor_obj=False)
# two_room_break_stop_goto_win_distr_obj_irrelevant_rule = TwoRoomEnv(break_stop_rule=False, obj1_pos="right_anywhere", obj2_pos="right_anywhere", distractor_obj=True, distractor_rule_block=True, irrelevant_rule_distractor=True)
# two_room_break_stop_goto_win = TwoRoomEnv(break_stop_rule=False, obj1_pos="right_anywhere", obj2_pos="right_anywhere", distractor_obj=False, distractor_rule_block=False)

# # Variants of TwoRoom MakeWin with various configurations
# two_room_make_win_distr_obj_rule = TwoRoomEnv(break_stop_rule=True, break_win_rule=True, obj1_pos="left_anywhere", obj2_pos="left_anywhere")
# two_room_make_win_distr_rule = TwoRoomEnv(break_stop_rule=True, break_win_rule=True, obj1_pos="left_anywhere", obj2_pos="left_anywhere", distractor_obj=False)
# two_room_make_win = TwoRoomEnv(break_stop_rule=True, break_win_rule=True, obj1_pos="left_anywhere", obj2_pos="left_anywhere", distractor_obj=False, distractor_rule_block=False)
# two_room_make_win_distr_obj_irrelevant_rule = TwoRoomEnv(break_stop_rule=True, break_win_rule=True, obj1_pos="left_anywhere", obj2_pos="left_anywhere", distractor_rule_block=True, irrelevant_rule_distractor=True)
# two_room_make_win_distr_obj = TwoRoomEnv(break_stop_rule=True, break_win_rule=True, obj1_pos="left_anywhere", obj2_pos="left_anywhere", distractor_rule_block=False)
# two_room_make_win_distr_win_rule = TwoRoomEnv(obj1_pos="left_anywhere", obj2_pos="left_anywhere", distractor_obj=True, distractor_win_rule=True, break_win_rule=True, break_stop_rule=True)

# # Anywhere Two Room
# two_room_maybe_break_stop_goto_win_distr_obj_rule= TwoRoomEnv(break_stop_rule=False, obj1_pos="anywhere", obj2_pos="anywhere")
# two_room_maybe_break_stop_goto_win = TwoRoomEnv(break_stop_rule=False, obj1_pos="anywhere", obj2_pos="anywhere", distractor_obj=False, distractor_rule_block=False)
# two_room_maybe_break_stop_goto_win_distr_obj = TwoRoomEnv(break_stop_rule=False, obj1_pos="anywhere", obj2_pos="anywhere", distractor_obj=True, distractor_rule_block=False)
# two_room_maybe_break_stop_goto_win_distr_rule = TwoRoomEnv(break_stop_rule=False, obj1_pos="anywhere", obj2_pos="anywhere", distractor_obj=False, distractor_rule_block=True)
# two_room_maybe_break_stop_goto_win_distr_obj_irrelevant_rule = TwoRoomEnv(break_stop_rule=False, obj1_pos="anywhere", obj2_pos="anywhere", distractor_obj=True, distractor_rule_block=True, irrelevant_rule_distractor=True)

# # TwoRoom MakeYou and MakeWallWin with specific configurations
# # the hardest levels!
# two_room_make_you_make_win = TwoRoomMakeYouEnv(break_win_rule=True)
# two_room_make_you = TwoRoomMakeYouEnv()
# two_room_make_wall_win = TwoRoomMakeWallWinEnv()

