from IPython.core.display import HTML, display

def dict_to_html(dictionary):
    html = '<ul>'
    for key, value in dictionary.iteritems():
        html += '<li><strong>{}</strong>: '.format(key)
        if isinstance(value, list):
            html += '<ul>'
            for item in value:
                html += '<li>{}</li>'.format(item)
            html += '</ul>'
        elif isinstance(value, dict):
            html += dict_to_html(value)
        else:
            try:
                html += '{}'.format(value)
            except UnicodeEncodeError:
                pass
        html += '</li>'
    html += '</ul>'
    return html
    
def print_dict(dictionary):
    html = dict_to_html(dictionary)
    display(HTML(html))
