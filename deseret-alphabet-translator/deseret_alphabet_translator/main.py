import os
import logging

import jinja2
import webapp2
from webapp2_extras import json

from english_to_deseret import EnglishToDeseret
from deseret_to_english import DeseretToEnglish

#logging.basicConfig(level=logging.DEBUG, format='[%(levelname)s] %(module)s %(funcName)s(): %(message)s')

JINJA_ENVIRONMENT = jinja2.Environment(
    loader=jinja2.FileSystemLoader(os.path.dirname(__file__)),
    extensions=['jinja2.ext.autoescape'],
    autoescape=True)


def get_globals():

    app = webapp2.get_app()
    d2e = app.registry.get('d2e')
    e2d = app.registry.get('e2d')
    if not d2e:
        logging.info("Globals were not initialized. Initializing now...")
        e2d = EnglishToDeseret()
        app.registry['e2d'] = e2d
        d2e = DeseretToEnglish()
        app.registry['d2e'] = d2e

    return {'d2e': d2e, 'e2d':e2d}


class MainPage(webapp2.RequestHandler):

    def post(self):
        g = get_globals()

        d2e = g['d2e'];
        english = self.request.params.get('english')

        #print "english = %s" % english
        if english:
            deseret = d2e.translate(english)
        else:
            english = ""
            deseret = ""

        template_values = {
            "english": english,
            "deseret": deseret
        }

        template = JINJA_ENVIRONMENT.get_template('index.html')
        self.response.write(template.render(template_values))


    def get(self):
        self.post()


class ApiPage(webapp2.RequestHandler):

    def post(self):
        template_values = {}
        template = JINJA_ENVIRONMENT.get_template('api.html')
        self.response.write(template.render(template_values))

    def get(self):
        self.post()

class WarmupHandler(webapp2.RequestHandler):

    def get(self):
        logging.info("Warming up...")
        get_globals()
        self.response.write("All warmed up!")

class JsonHandler(webapp2.RequestHandler):

    def post(self):

        json_obj = json.decode(self.request.body)
        english = json_obj.get('english', None)
        g = get_globals()
        self.response.content_type = 'application/json; charset=utf-8'

        if not english is None:
            e2d = g['e2d']
            #logging.info("Translating '%s'..." % english)
            deseret = e2d.translate(english)
            #logging.info("Translation: '%s'" % deseret)
            obj = {
                'deseret': deseret
            }
            self.response.write(json.encode(obj))

        else:
            deseret = json_obj.get('deseret', None)
            if not deseret is None:
                d2e = g['d2e']
                english = d2e.translate(deseret)
                obj = {
                    'english': english
                }
                self.response.write(json.encode(obj))
            else:
                self.abort(500, detail="Either 'english' or 'deseret' parameter must be provided.")



def handle_error(request, response, exception):
    logging.exception(exception)
    error_code = 500
    error_description = str(exception)
    if isinstance(exception, webapp2.HTTPException):
        error_code = exception.code
        error_description = exception.explanation

    response.status = error_code

    if request.path.startswith('/json'):
        response.headers.add_header('Content-Type', 'application/json')
        result = {
            'status': 'error',
            'status_code': error_code,
            'error_message': error_description,
          }
        response.write(json.encode(result))
    else:
        response.write(error_description)


application = webapp2.WSGIApplication([
    ('/', MainPage),
    ('/_ah/warmup', WarmupHandler),
    ('/json/translation', JsonHandler),
    ('/api', ApiPage)
], debug=True)
application.error_handlers[404] = handle_error
application.error_handlers[400] = handle_error
application.error_handlers[500] = handle_error


