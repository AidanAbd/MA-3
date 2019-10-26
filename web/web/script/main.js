import {dom} from '../util/webDom';

import * as ws from './ws';
import {uppy} from './uppy';

ws.useUppy(uppy);

document.addEventListener('DOMContentLoaded', () => {

});

const classNameField = document.getElementById('class-name-in');
const nextBtn = document.getElementById('next-btn');
const submitBtn = document.getElementById('train-btn');
const classTableBody = document.getElementById('class-table-body');
const errorAlert = document.getElementById('error-alert');

let classnum = 1;

nextBtn.addEventListener('click', () => {
  const fs = uppy.getFiles();

  if (classNameField.value === '') {
    errorAlert.innerText = 'No class label specified';
    $('#error-alert').removeClass('d-none');
    return;
  }

  if (fs.length === 0) {
    errorAlert.innerText = 'No files uploaded';
    $('#error-alert').removeClass('d-none');
    return;
  }

  const us = uppy.getState();
  console.log(us);
  if (
    Object.keys(us.currentUploads).length !== 0 ||
    us.totalProgress !== 100
  ) {
    errorAlert.innerText = 'File upload incomplete';
    $('#error-alert').removeClass('d-none');
    return;
  }

  $('#error-alert').addClass('d-none');
  ws.setWorkingsetClassName(classNameField.value);

  const closeBtn = <button type='button' class='close'>&times;</button>.el;
  const row =
    <tr>
      <td>{closeBtn}</td>
      <td>{classNameField.value}</td>
      <td>{String(fs.length)}</td>
      <td>10 MB</td>
    </tr>.el;

  classTableBody.appendChild(row);
  closeBtn.addEventListener('click', () => {
    classTableBody.removeChild(row);
  });

  classNameField.placeholder = `class-${++classnum}`;
  classNameField.value = '';
  uppy.reset();
});

submitBtn.addEventListener('click', () => {
  ws.startTraining();
});


