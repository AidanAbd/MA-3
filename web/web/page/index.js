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
        <title>Flash Computer Vision</title>
        <meta name='description' content=''/>
        <meta name='viewport' content='width=device-width, initial-scale=1, shrink-to-fit=no'/>

        <link rel='apple-touch-icon' sizes='180x180' href='/apple-touch-icon.png'/>
        <link rel='icon' type='image/png' sizes='32x32' href='/favicon-32x32.png'/>
        <link rel='icon' type='image/png' sizes='16x16' href='/favicon-16x16.png'/>
        <link rel='manifest' href='/site.webmanifest'/>
        <link rel='mask-icon' href='/safari-pinned-tab.svg' color='#d1a830'/>
        <meta name='msapplication-TileColor' content='#fec00a'/>
        <meta name='theme-color' content='#fec00a'/>

        <link rel='stylesheet' href='https://stackpath.bootstrapcdn.com/bootstrap/4.3.1/css/bootstrap.min.css' integrity='sha384-ggOyR0iXCbMQv3Xipma34MD+dH/1fQ784/j6cY/iJTQUOhcWr7x9JvoRxT2MZw1T' crossorigin='anonymous'/>

        <link rel='stylesheet' href={require('@uppy/core/dist/style.css')}/>
        <link rel='stylesheet' href={require('@uppy/webcam/dist/style.css')}/>
        <link rel='stylesheet' href={require('@uppy/dashboard/dist/style.css')}/>
        <link rel='stylesheet' href={require('../style/main.styl')}/>

        <meta name='theme-color' content='#fafafa'/>
      </head>
      <body>
        {{str: ieWarning.join(pretty ? '\n' : '')}}
        <div class='container-fluid'>
          <div class='row'>
            <div class='col-md-3'>
              <img src={require('../res/img/sidebar.svg')}/>
            </div>
            <div class='col-md-6'>
              <div class='row col-12' id='training-label-box'>
                <h1>Training</h1>
                <hr class='w-100'/>
              </div>
              <div id='sample-collector'>
                <div class='row'>
                  <div class='col-12'>
                    <h2>Add Samples</h2>
                  </div>
                </div>
                <div class='row'>
                  <div class='col-md-5 order-2 order-md-1'>
                    <label>Class label:</label>
                    <div class='input-group'>
                      <input id='class-name-in' class='form-control' type='text' placeholder='class-1'/>
                    </div>
                    <div class='height-1em'/>
                    <button type='button' class='w-100 btn btn-primary' id='next-btn'>Add to set</button>
                    <div class='height-1em'/>
                    <div class='alert alert-danger d-none' id='add-class-error-alert'/>
                    <table class='table table-sm'>
                      <thead>
                        <tr>
                          <th></th>
                          <th>Class</th>
                          <th>#</th>
                          <th>Total Size</th>
                        </tr>
                      </thead>
                      <tbody class='overflow-auto' id='class-table-body'/>
                    </table>
                  </div>
                  <div class='col-md-7 order-1 order-md-2'>
                    <div id='drag-drop-area'/>
                  </div>
                </div>
                <div class='row'>
                  <div class='col-12'>
                    <hr class='w-100'/>
                  </div>
                </div>
              </div>
              <div class='row' id='training-progress-box' class='d-none'>
                <div class='col-12'>
                  <div class='progress height-3em'>
                    <div class='progress-bar' id='training-progress-bar'/>
                  </div>
                </div>
              </div>
              <div id='start-training-box'>
                <div class='row height-1em'/>
                <div class='row justify-content-center'>
                  <button type='button' class='btn btn-primary' id='train-btn'>Start training</button>
                </div>
                <div class='row height-1em'/>
                <div class='row justify-content-center'>
                  <div class='alert alert-danger d-none' id='start-training-error-alert'/>
                </div>
                <div class='row height-1em'/>
              </div>
              <div id='inference-box' class='d-none'>
                <div class='row'>
                  <div class='col-12'>
                    <h1>Inference</h1>
                    <hr/>
                  </div>
                </div>
                <div class='row height-1em'/>
                <div class='row'>
                  <div class='col-12'>
                    <div id='drag-drop-area-inference'/>
                  </div>
                </div>
                <div class='row' id='inference-results-box'>
                </div>
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
