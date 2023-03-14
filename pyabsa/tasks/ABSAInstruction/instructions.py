class InstructionsHandler:
    def __init__(self):
        self.ate = {}
        self.atsc = {}
        self.joint = {}
        self.multitask = {}

    def load_instruction_set1(self):
        # self.multitask['bos_instruct1'] = """Definition: The output will be the aspects, opinions, sentiment polarities
        # and aspect categories (both implicit and explicit). In cases where there are no aspects (or aspects, opinions,
        # and aspect categories) the output should be NULL.
        # Positive example 1-
        # input: I charge it at night and skip taking the cord with me because of the good battery life.
        # output: aspect:battery life|opinion:good|sentiment:positive|category:POWER_SUPPLY#GENERAL, aspect:cord|opinion:NULL|sentiment:positive|category:POWER_SUPPLY#GENERAL
        # Positive example 2-
        # input: the keyboard is a delight with large , nearly flush , keys set to just the right level of resistance and sensitivity .
        # output: aspect:keyboard|opinion:large|sentiment:positive|category:LAPTOP#GENERAL, keys:right level of resistance and sensitivity:positive:LAPTOP#GENERAL
        # Now complete the following example-
        # input: """

        self.multitask[
            "bos_instruct1"
        ] = """Definition: The output will be the aspects, opinions, sentiment polarities
        and aspect categories (both implicit and explicit). In cases where there are no aspects (or aspects, opinions,
        and aspect categories) the output should be NULL.
        Positive example 1-
        input: I charge it at night and skip taking the cord with me because of the good battery life.
        output: battery life|good|positive|POWER_SUPPLY#GENERAL, cord|NULL|positive|POWER_SUPPLY#GENERAL
        Positive example 2-
        input: the keyboard is a delight with large , nearly flush , keys set to just the right level of resistance and sensitivity .
        output: keyboard|large|positive|LAPTOP#GENERAL, keys|right level of resistance and sensitivity|positive|LAPTOP#GENERAL
        Now complete the following example-
        input: """

        self.multitask[
            "bos_instruct2"
        ] = """Definition: The output will be the aspects (both implicit and explicit) and the aspects sentiment polarity. In cases where there are no aspects the output should be noaspectterm:none.
        Positive example 1-
        input: With the great variety on the menu , I eat here often and never get bored.
        output: menu|great variety|positive|RESTAURANT#GENERAL, food|great variety|positive|RESTAURANT#GENERAL
        Positive example 2- 
        input: Great food, good size menu, great service and an unpretensious setting.
        output: food|great|positive|RESTAURANT#GENERAL, menu|good size|positive|RESTAURANT#GENERAL, service|great|positive|RESTAURANT#GENERAL, setting|unpretensious|positive|RESTAURANT#GENERAL
        Now complete the following example-
        input: """

        self.multitask[
            "bos_instruct3"
        ] = """Definition: The output will be the aspects, opinions, sentiment polarities
        and aspect categories (both implicit and explicit). In cases where there are no aspects (or aspects, opinions,
        and aspect categories) the output should be NULL.
        Positive example 1-
        input: I charge it at night and skip taking the cord with me because of the good battery life.
        output: aspect:battery life|opinion:good|polarity:positive|category:POWER_SUPPLY#GENERAL, aspect:cord|opinion:NULL|polarity:positive|category:POWER_SUPPLY#GENERAL
        Positive example 2-
        input: Great food, good size menu, great service and an unpretensious setting.
        output: aspect:food|opinion:great|polarity:positive|category:RESTAURANT#GENERAL, aspect:menu|opinion:good size|polarity:positive|category:RESTAURANT#GENERAL, aspect:service|opinion:great|polarity:positive|category:RESTAURANT#GENERAL, aspect:setting|opinion:unpretensious|sentiment:positive|category:RESTAURANT#GENERAL
        Now complete the following example-
        input: """

        # self.multitask['bos_instruct3'] = """
        # Example #1
        # input: I charge it at night and skip taking the cord with me because of the good battery life.
        # output: aspect:battery life|opinion:good|polarity:positive|category:POWER_SUPPLY#GENERAL, aspect:cord|opinion:NULL|polarity:positive|category:POWER_SUPPLY#GENERAL
        # Example #2
        # input: Great food, good size menu, great service.
        # output: aspect:food|opinion:great|polarity:positive|category:RESTAURANT#GENERAL, aspect:menu|opinion:good size|polarity:positive|category:RESTAURANT#GENERAL, aspect:service|opinion:great|polarity:positive|category:RESTAURANT#GENERAL
        # Now complete the following example-
        # input: """

        self.multitask["delim_instruct"] = ""

        self.multitask["eos_instruct"] = " \noutput:"
