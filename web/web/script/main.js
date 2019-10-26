import prettyBytes from 'pretty-bytes';

import {dom} from '../util/webDom';

import * as ws from './ws';
import {uppy} from './uppy';

const addClassErrorAlert = document.getElementById('add-class-error-alert');
const startTrainingErrorAlert = document.getElementById('start-training-error-alert');

const showAddClassError = (err) => {
  addClassErrorAlert.innerText = err;
  $('#add-class-error-alert').removeClass('d-none');
};

const showStartTrainingError = (err) => {
  startTrainingErrorAlert.innerText = err;
  $('#start-training-error-alert').removeClass('d-none');
};

try {
  ws.useUppy(uppy);

  document.addEventListener('DOMContentLoaded', () => {

  });

  const classNameField = document.getElementById('class-name-in');
  const nextBtn = document.getElementById('next-btn');
  const submitBtn = document.getElementById('train-btn');
  const classTableBody = document.getElementById('class-table-body');
  const sampleCollector = document.getElementById('sample-collector');
  const trainingProgressBar = document.getElementById('training-progress-bar');
  const trainingProgressBox = document.getElementById('training-progress-box');
  const startTrainingBox = document.getElementById('start-training-box');

  let classnum = 1;
  let classAmount = 0;

  nextBtn.addEventListener('click', () => {
    try {
      const fs = uppy.getFiles();

      if (classNameField.value === '') {
        showAddClassError('No class label specified');
        return;
      }

      if (fs.length === 0) {
        showAddClassError('No files uploaded');
        return;
      }

      const us = uppy.getState();
      if (
        Object.keys(us.currentUploads).length !== 0 ||
        us.totalProgress !== 100
      ) {
        showAddClassError('File upload incomplete');
        return;
      }

      $('#add-class-error-alert').addClass('d-none');
      ws.setWorkingsetClassName(classNameField.value);

      let totalSize = 0;
      for (const f of fs) {
        totalSize += f.size;
      }

      const closeBtn = <button type='button' class='close'>&times;</button>.el;
      const className = classNameField.value;
      const row =
        <tr>
          <td>{closeBtn}</td>
          <td>{className}</td>
          <td>{String(fs.length)}</td>
          <td>{prettyBytes(totalSize)}</td>
        </tr>.el;

      classTableBody.appendChild(row);
      closeBtn.addEventListener('click', () => {
        classTableBody.removeChild(row);
        ws.removeClass(className);
        --classAmount;
      });

      classNameField.placeholder = `class-${++classnum}`;
      classNameField.value = '';
      uppy.reset();
      ++classAmount;
    }
    catch (e) {
      showAddClassError('Internal error');
      showStartTrainingError('Internal error');

      throw e;
    }
  });

  submitBtn.addEventListener('click', () => {
    try {
      if (classAmount < 2) {
        showStartTrainingError('Need at least 2 classes to start training');
        return;
      }

      ws.setProgressCb((completeness) => {
        const str = Math.round(completeness*100)+'%';
        trainingProgressBar.style.width = str;
        trainingProgressBar.innerText = str;
      });
      ws.startTraining();

      sampleCollector.className = 'd-none';
      trainingProgressBox.className = '';
      startTrainingBox.className = 'd-none';
    }
    catch (e) {
      showAddClassError('Internal error');
      showStartTrainingError('Internal error');

      throw e;
    }
  });
}
catch(e) {
  showAddClassError('Internal error');
  showStartTrainingError('Internal error');

  throw e;
}
