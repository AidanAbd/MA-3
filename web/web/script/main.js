import prettyBytes from 'pretty-bytes';

import {dom} from '../util/webDom';

import * as ws from './ws';
import {setupUppyFor} from './uppy';

let uppy = setupUppyFor('drag-drop-area', 'upload-samples');

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

  const classNameField = document.getElementById('class-name-in');
  const nextBtn = document.getElementById('next-btn');
  const submitBtn = document.getElementById('train-btn');
  const classTableBody = document.getElementById('class-table-body');
  const sampleCollector = document.getElementById('sample-collector');
  const trainingProgressBar = document.getElementById('training-progress-bar');
  const trainingProgressBox = document.getElementById('training-progress-box');
  const startTrainingBox = document.getElementById('start-training-box');
  const inferenceBox = document.getElementById('inference-box');
  const inferenceResultsBox = document.getElementById('inference-results-box');
  const trainingLabelBox = document.getElementById('training-label-box');

  let classnum = 1;
  let classAmount = 0;

  classNameField.addEventListener('keyup', () => {
    classNameField.value = classNameField.value.replace(/ /g, '');
  });

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

        if (completeness === 1) {
          trainingProgressBox.className = 'd-none';
          inferenceBox.className = '';
          trainingLabelBox.className = 'd-none';

          let oldMeta = uppy.getState().meta;
          uppy = setupUppyFor('drag-drop-area-inference', 'upload-inference-image');
          uppy.setMeta(oldMeta);

          const filenameMap = new Map();
          const filenameToElementsMap = new Map();
          uppy.on('file-added', (f) => {
            filenameMap.set(f.name, f.id);
          });
          uppy.on('file-removed', (f) => {
            for (const el of filenameToElementsMap.get(f.name)) {
              inferenceResultsBox.removeChild(el);
            }

            filenameMap.delete(f.name);
            filenameToElementsMap.delete(f.name);
          });

          ws.setInferenceCB((filename, prediction, accuracy) => {
            const file = uppy.getFile(filenameMap.get(filename));
            if (file == null)
              return;

            if (filenameToElementsMap.get(filename) != null)
              return;

            const pad1 = <div class='col-sm-05'/>.el;
            const card = <div class='col-sm-3 card inference-card'>
              <div class='square'>
                <div class='inference-image-container'>
                  <img class='card-img-top inference-image' src={file.preview}/>
                </div>
              </div>
              <div><hr/></div>
              <div>
                <div class='text-center font-weight-bold text-monospace'>{filename}</div>
                <div class='text-center'>{prediction} ({Math.round(accuracy*100)+'%'})</div>
                <div class='height-1em'/>
              </div>
            </div>.el;
            const pad2 = <div class='col-sm-05'/>.el;

            filenameToElementsMap.set(filename, [pad1, card, pad2]);

            inferenceResultsBox.appendChild(pad1);
            inferenceResultsBox.appendChild(card);
            inferenceResultsBox.appendChild(pad2);
          });
        }
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
