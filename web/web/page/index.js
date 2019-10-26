import {dom} from '../util/dom';

import webpackMode from '@@webpackMode';
const pretty = webpackMode === 'development';

const ieWarning = [
  '<!--[if IE]>',
  '  <p class="browserupgrade">',
  '   You are using an <strong>outdated</strong> browser.',
  '   Please <a href="https://browsehappy.com/">upgrade your browser</a>',
  '   to improve your experience and security.',
  '  </p>',
  '<![endif]-->',
  ''
];

export default () => {
  return '<!doctype html>' + (pretty ? '\n' : '')+
    <html lang='en'>
      <head>
        <meta charset='utf-8'/>
        <title>ParserGen</title>
        <meta name='description' content=''/>
        <meta name='viewport' content='width=device-width, initial-scale=1, shrink-to-fit=no'/>

        {/*<link rel="manifest" href="site.webmanifest"/>
        <link rel="apple-touch-icon" href="icon.png"/>*/}

        <link rel='stylesheet' href='https://stackpath.bootstrapcdn.com/bootstrap/4.3.1/css/bootstrap.min.css' integrity='sha384-ggOyR0iXCbMQv3Xipma34MD+dH/1fQ784/j6cY/iJTQUOhcWr7x9JvoRxT2MZw1T' crossorigin='anonymous'/>
        <link href='https://transloadit.edgly.net/releases/uppy/v1.5.2/uppy.min.css' rel='stylesheet'/>

        <meta name='theme-color' content='#fafafa'/>
      </head>
      <body>
        {{str: ieWarning.join(pretty ? '\n' : '')}}
        <div class='container-fluid'>
          <div class='row justify-content-center'>
            <div class='col-sm-6'>
              <h1>Test and all</h1>
              <div id='drag-drop-area'/>
            </div>
          </div>
        </div>

        <script src='https://code.jquery.com/jquery-3.3.1.slim.min.js' integrity='sha384-q8i/X+965DzO0rT7abK41JStQIAqVgRVzpbzo5smXKp4YfRvH+8abtTE1Pi6jizo' crossorigin='anonymous'/>
        <script src='https://cdnjs.cloudflare.com/ajax/libs/popper.js/1.14.7/umd/popper.min.js' integrity='sha384-UO2eT0CpHqdSJQ6hJty5KVphtPhzWj9WO1clHTMGa3JDZwrnQq4sF86dIHNDz0W1' crossorigin='anonymous'/>
        <script src='https://stackpath.bootstrapcdn.com/bootstrap/4.3.1/js/bootstrap.min.js' integrity='sha384-JjSmVgyd0p3pXB1rRibZUAYoIIy6OrQ6VrjIEaFf/nJGzIxFDsf4x0xIM+B07jRM' crossorigin='anonymous'/>

        <script src='https://transloadit.edgly.net/releases/uppy/v1.5.2/uppy.min.js'/>
      </body>
    </html>.str;
};
