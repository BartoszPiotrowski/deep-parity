


def parse(term):
    '''
    term is arithmetical expression with + and - only, eg. 1-2+3-0+1.
    We parse always from "right to left". TODO: randomize it.
    '''
    if len(term) == 1:
        return term # TODO or [term]?
    else:
        return [term[-2], [parse(term[:-2]), term[-1]]]

