## get icd9 info
def get_icd9info(icd9code,type='disease',includeShortName=True):
    """Get info related to icd9code

    Info:
        https://clinicaltables.nlm.nih.gov/

    Reference:
        https://clinicaltables.nlm.nih.gov/apidoc/icd9cm_dx/v3/doc.html

    Args:
        icd9code (str): Code to query.
        type (option): `disease` or `procedure`.
        includeShortName: Also include the short name in output.
    """
    import requests, json
    if type == 'disease':
        url = 'https://clinicaltables.nlm.nih.gov/api/icd9cm_dx/v3/search?terms={term}'
    elif type == 'procedure':
        url = 'https://clinicaltables.nlm.nih.gov/api/icd9cm_sg/v3/search?terms={term}'
    else: raise Exception('Wrong type specified, should be disease or procedure, not',type)
    if includeShortName: url+='&ef=short_name'
    r = requests.get(url.format(term = icd9code))
    return json.loads(r.content)
