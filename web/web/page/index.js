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

        <link rel='stylesheet' href={require('@uppy/core/dist/style.css')}/>
        <link rel='stylesheet' href={require('@uppy/dashboard/dist/style.css')}/>

        <meta name='theme-color' content='#fafafa'/>
      </head>
      <body>
        {{str: ieWarning.join(pretty ? '\n' : '')}}
        <div class='container-fluid'>
          <div class='row justify-content-center'>
            <div class='col-sm-6'>
              <h1>Training</h1>
              <hr/>
              <h2>Add Samples</h2>
            </div>
          </div>
          <div class='row justify-content-center'>
            <div class='col-sm-2'>
              <label>Class label:</label>
              <div class='input-group'>
                <input id='class-name-in' class='form-control' type='text' placeholder='class-1'/>
              </div>
              <button type='button' class='btn btn-primary' id='next-btn'>Next</button>
            </div>
            <div class='col-sm-4'>
              <div id='drag-drop-area'/>
            </div>
          </div>
        </div>
        <div class='modal fade' id='no-class-name-modal'>
          <div class='modal-dialog'>
            <div class='modal-content'>
              <div class='modal-header'>
                <h5 class='modal-title'>Pl</h5>
                <button type='button' class='close' data-dismiss='modal'>&times;</button>
              </div>
              <div class='modal-body'>
                <label for='class-name'>Class label:</label>
                <div class='input-group'>
                  <input id='class-name' class='form-control' type='text' placeholder='class-1'/>
                </div>
                <button type='button' class='btn btn-primary' id='no-class-name-modal-submit'>Submit</button>
              </div>
            </div>
          </div>
        </div>

        <script src='https://code.jquery.com/jquery-3.3.1.slim.min.js' integrity='sha384-q8i/X+965DzO0rT7abK41JStQIAqVgRVzpbzo5smXKp4YfRvH+8abtTE1Pi6jizo' crossorigin='anonymous'/>
        <script src='https://cdnjs.cloudflare.com/ajax/libs/popper.js/1.14.7/umd/popper.min.js' integrity='sha384-UO2eT0CpHqdSJQ6hJty5KVphtPhzWj9WO1clHTMGa3JDZwrnQq4sF86dIHNDz0W1' crossorigin='anonymous'/>
        <script src='https://stackpath.bootstrapcdn.com/bootstrap/4.3.1/js/bootstrap.min.js' integrity='sha384-JjSmVgyd0p3pXB1rRibZUAYoIIy6OrQ6VrjIEaFf/nJGzIxFDsf4x0xIM+B07jRM' crossorigin='anonymous'/>
      </body>
    </html>.str;
};
