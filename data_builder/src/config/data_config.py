CONSTRAINT_TYPES = {'pad': 0, 'positive': 1, 'negative': 2, 'optional': 3}

# SIMPLE_QAS_TEMPLATES = ['How much {p} does {s} have?',
#                         'How much {p} does {s} contain?',
#                         'How much {p} does {s} consist of?',
#                         'How much {p} is in {s}?',
#                         'What is {p} value in {s}?',
#                         'How much {p} is there in {s}?']

# COMPARISION_QAS_TEMPLATES = [(True, '{s1} or {s2}, which has more {p}?'),
#                             (False, '{s1} or {s2}, which has less {p}?'),
#                             (True, 'Which has more {p}, {s1} or {s2}?'),
#                             (False, 'Between {s1} and {s2}, which one has less {p}?'),
#                             (True, '{s1} or {s2}, which one has higher amount of {p}?'),
#                             (False, '{s1} or {s2}, which one has lower amount of {p}?'),
#                             (True, 'Which contains more {p}, {s1} or {s2}?'),
#                             (True, '{s1} or {s2}, which contains more {p}?'),
#                             (False, '{s1} or {s2}, which contains less {p}?'),
#                             (False, 'Between {s1} and {s2}, which contains less {p}?'),
#                             (True, 'Between {s1} and {s2}, which one consists of larger amount of {p}?'),
#                             (True, '{s1} or {s2}, which one consists of higher amount of {p}?'),
#                             (False, '{s1} or {s2}, which one consists of lower amount of {p}?'),
#                             (False, 'Which consists of lower amount of {p}, {s1} or {s2}?'),
#                             ]

CONSTRAINT_QAS_TEMPLATES = ['What are {tag} recipes that contain {in_list}?',
                            'What are {tag} dishes that have ingredients {in_list}?',
                            'What are {tag} dishes which consist of {in_list}?',
                            'What are {tag} recipes that consist of ingredients {in_list}?',
                            'What are {tag} recipes which have ingredients {in_list}?',
                            'What {tag} dishes have {in_list}?',
                            'What {tag} dishes contain ingredients {in_list}?',
                            'What {tag} recipes can I cook with {in_list}?',
                            'What {tag} dishes can I make with {in_list}?',
                            'Suggest {tag} dishes that contain {in_list}?',
                            'Recommend {tag} recipes which have ingredients {in_list}?',
                            'Can you suggest {tag} recipes that consist of {in_list}?',
                            'Could you recommend {tag} dishes which contain {in_list}?',
                            # Add
                            'What {tag} dishes can I take that contain {in_list}?',
                            ]

CONSTRAINT_QAS_TEMPLATES_NEG = [
                            'What are {tag} recipes that do not contain {in_list}?',
                            'What are {tag} dishes that do not have ingredients {in_list}?',
                            "What are {tag} dishes which don't consist of {in_list}?",
                            'What are {tag} recipes that do not consist of ingredients {in_list}?',
                            'What are {tag} recipes which do not have ingredients {in_list}?',
                            "What {tag} dishes don't have {in_list}?",
                            'What {tag} dishes do not contain ingredients {in_list}?',
                            'What {tag} recipes can I cook without {in_list}?',
                            'What {tag} dishes can I make without {in_list}?',
                            'Suggest {tag} dishes that do not contain {in_list}?',
                            'Recommend {tag} recipes which do not have ingredients {in_list}?',
                            'Can you suggest {tag} recipes that do not consist of {in_list}?',
                            'Could you recommend {tag} dishes which do not contain {in_list}?',
                            # Add
                            'What {tag} dishes can I make that do not contain {in_list}?',
                            ]

CONSTRAINT_QAS_TEMPLATES_LIMITS_POS = [
                            'What are {limit} {nutrient} {tag} recipes that contain {in_list}?',
                            'What are {limit} {nutrient} {tag} dishes that have ingredients {in_list}?',
                            'What are {limit} {nutrient} {tag} dishes which consist of {in_list}?',
                            'What are {tag} recipes that consist of ingredients {in_list} and are {limit} {nutrient}?',
                            'What are {limit} {nutrient} {tag} recipes which have ingredients {in_list}?',
                            'What {tag} dishes have {in_list} and are {limit} {nutrient}?',
                            'What {limit} {nutrient} {tag} dishes contain ingredients {in_list}?',
                            'What {limit} {nutrient} {tag} recipes can I cook with {in_list}?',
                            'What {limit} {nutrient} {tag} dishes can I make with {in_list}?',
                            'Suggest {limit} {nutrient} {tag} dishes that contain {in_list}?',
                            'Recommend {limit} {nutrient} {tag} recipes which have ingredients {in_list}?',
                            'Can you suggest {tag} recipes that consist of {in_list} and are {limit} {nutrient}?',
                            'Could you recommend {limit} {nutrient} {tag} dishes which contain {in_list}?',
                            'What {limit} {nutrient} {tag} dishes can I take that contain {in_list}?',
                            ]
                            #neg
CONSTRAINT_QAS_TEMPLATES_LIMITS_NEG = [
                            'What are {limit} {nutrient} {tag} recipes that do not contain {in_list}?',
                            'What are {limit} {nutrient} {tag} dishes that do not have ingredients {in_list}?',
                            "What are {limit} {nutrient} {tag} dishes which don't consist of {in_list}?",
                            'What are {tag} recipes that do not consist of ingredients {in_list} and are {limit} {nutrient}?',
                            'What are {limit} {nutrient} {tag} recipes which do not have ingredients {in_list}?',
                            "What {tag} dishes don't have {in_list} and are {limit} {nutrient}?",
                            'What {limit} {nutrient} {tag} dishes do not contain ingredients {in_list}?',
                            'What {limit} {nutrient} {tag} recipes can I cook without {in_list}?',
                            'What {limit} {nutrient} {tag} dishes can I make without {in_list}?',
                            'Suggest {limit} {nutrient} {tag} dishes that do not contain {in_list}?',
                            'Recommend {limit} {nutrient} {tag} recipes which do not have ingredients {in_list}?',
                            'Can you suggest {tag} recipes that do not consist of {in_list} and are {limit} {nutrient}?',
                            'Could you recommend {limit} {nutrient} {tag} dishes which do not contain {in_list}?',
                            'What {limit} {nutrient} {tag} dishes can I make that do not contain {in_list}?',
                            ]

LIMIT_NUTRIENT_VALUES = {
        'protein': #https://www.webmd.com/food-recipes/protein
            {'low':
                 {'lower':0,
                  'upper':15.33}, #46g per day div by 3
             'medium':
                 {'lower':15.33,
                  'upper':18.67}, #56g per day div by 3
             'high':
                 {'lower':18.67,
                  'upper':10000}
             },
        'carbohydrates': #https://www.health.com/condition/type-2-diabetes/how-to-count-carbs-in-10-common-foods
            {'low':
                 {'lower':0,
                  'upper':45},
             'medium':
                 {'lower':45,
                  'upper':60},
             'high':
                 {'lower':60,
                  'upper':10000}
              },
        'fat': #https://my.clevelandclinic.org/health/articles/11208-fat-what-you-need-to-know
            {'low':
                 {'lower':0,
                  'upper':14.67},
             'medium':
                 {'lower':14.67,
                  'upper':25.67}, #44-77g per day div by 3
             'high':
                 {'lower':25.67,
                  'upper':10000}
             }
        }


GUIDELINE_DIRECTIVES = [
{'calories' :
    {'unit': 'calories',
     'meal' :
       {'type': 'range',
       'lower' : '800',
       'upper': '1500'},
    'daily total' : '1500'}
    },


{'calories' :
    {'unit': 'calories',
     'meal' :
       {'type': 'range',
       'lower' : '100',
       'upper': '800'},
    'daily total' : '800'}
    },


{'fat':
      {'percentage': 'calories',
      'multiplier': 9,
      'type': 'range',
      'meal': {
          'lower': '20',
          'upper': '35'}
      }
  },

    {'saturated fat':
      {'percentage': 'calories',
      'multiplier': 9,
      'type': 'range',
      'meal': {
          'lower': '5',
          'upper': '20'}
      }
  },

  {'saturated fat':
      {'percentage': 'calories',
      'multiplier': 9,
      'type': 'range',
      'meal': {
          'lower': '10',
          'upper': '35'}
      }
  },

{'carbohydrates' :
    {'unit': 'g',
     'meal' :
       {'type': 'range',
       'lower' : '5',
       'upper': '30'},
    'daily total' : '30'}
    },


{'carbohydrates':
      {'percentage': 'calories',
      'multiplier': 4,
      'type': 'range',
      'meal': {
          'lower': '0',
          'upper': '45'}
      }
  },


{'carbohydrates' :
    {'unit': 'g',
     'meal' :
       {'type': 'range',
       'lower' : '15',
       'upper': '50'},
    'daily total' : '150'}
    },

  {'carbohydrates':
      {'percentage': 'calories',
      'multiplier': 4,
      'type': 'range',
      'meal': {
          'lower': '35',
          'upper': '65'}
      }
  },

    {'carbohydrates':
      {'percentage': 'calories',
      'multiplier': 4,
      'type': 'range',
      'meal': {
          'lower': '10',
          'upper': '40'}
      }
  },

  {'protein' :
    {'unit': 'g',
     'meal' :
       {'type': 'range',
       'lower' : '5',
       'upper': '30'},
    'daily total' : '60'}
    },

    {'protein' :
    {'unit': 'g',
     'meal' :
       {'type': 'range',
       'lower' : '15',
       'upper': '40'},
    'daily total' : '60'}
    },


  {'protein':
      {'percentage': 'calories',
      'multiplier': 4,
      'type': 'range',
      'meal': {
          'lower': '10',
          'upper': '25'}
      }
  },

  {'protein':
      {'percentage': 'calories',
      'multiplier': 4,
      'type': 'range',
      'meal': {
          'lower': '5',
          'upper': '20'}
      }
  },
  {'sugar' :
    {'unit': 'g',
     'meal' :
       {'type': 'range',
       'lower' : '5',
       'upper': '30'},
    'daily total' : '150'}
    },


    {'sugar' :
    {'unit': 'g',
     'meal' :
       {'type': 'range',
       'lower' : '10',
       'upper': '35'},
    'daily total' : '150'}
    }
  ]
