export const handleWs = (ws) => {
  let packetCounter = 0;

  const send = (type, obj)=>{
    if (obj.type != null)
      throw new Error('Packet payload cannot specify packet type.');

    ws.send(JSON.stringify(Object.assign(obj, {
      type: type,
      id: packetCounter++
    })));
  };
  const sendAck = (obj)=>{
    send('ack', {id: obj.id});
  };

  const errorMessages = {
    'bad-packet-json': 'Received packet that is invalid JSON.',
    'null-packet': 'Received a null packet.',
    'no-such-packet': 'Received packet with invalid type: "${packet}".',
    'no-such-id': 'Received a ${packet_type} for invalid packet with id ${id}.',
    'missing-protocol-field': 'Received a ${packet_type} without the required protocol field ${field_name}.',
    'missing-body-field': 'Received a ${packet_type} without the required body field ${field_name}.',
    'field-out-of-range': 'Field ${field_name} is unexpectedly not in range ${range_string}.',
    'field-of-wrong-type': 'Field ${field_name} is of invalid type, expected a ${expected_type}.',
    'bug': 'Something went wrong on the receiving side. This is a bug.',
    'wrong-pong': 'Received a pong with the wrong message. Expected: "${message}", not "${wrong_message}".',
    'path-does-not-exist': '${path} unexpectedly does not exist.',
    'unexpected-empty-string': 'Field ${field_name} is unexpectedly empty.',
    'unexpected-not-directory': '${path} unexpectedly is not a directory.',
    'unexpected-not-file': '${path} unexpectedly is not a file.',
    'unexpected-not-file-or-directory': '${path} unexpectedly is not a file or a directory.',
    'unexpected-class': '"${class}" is not one of previously specified classes.'
  };
  const sendError = (cause, code, etc)=>{
    send('error', {
      id: cause,
      code: code,
      etc: etc == null ? '' : etc
    });
  };

  ws.on('message', (data) => {
    let obj = null;
    try {
      obj = JSON.parse(data);
    }
    catch (e) {
      sendError('bad-json');
      return;
    }
    if (obj == null) {
      sendError('null-packet');
      return;
    }

    if (obj.type == null) {
      sendError('missing-protocol-field', 'type');
      return;
    }
    if (obj.id == null) {
      sendError('missing-protocol-field', 'id');
      return;
    }

    const requireBodyField = (packet, field) => {
      if (packet.data == null) {
        sendError('missing-protocol-field', 'data');
        return false;
      }

      if (packet.data[field] == null) {
        sendError('missing-body-field', field);
        return false;
      }

      return true;
    };

    if (obj.type === 'ack') {
      return;
    }
    else if (obj.type === 'error') {
      if (!requireBodyField(obj, 'id')) return;
      if (!requireBodyField(obj, 'code')) return;
      if (!requireBodyField(obj, 'etc')) return;

      let msg = `Packet ${obj.data.id} caused ${obj.data.code}`;

      if (obj.etc !== '')
        msg += ` "${obj.data.etc}"`;

      console.error(msg);
      return;
    }
    else if (obj.type === 'ping') {
      if (!requireBodyField(obj, 'message')) return;

      send('pong', {id: obj.id, message: obj.data.message});
      return;
    }
    else if (obj.type === 'pong') {
      if (!requireBodyField(obj, 'message')) return;
      console.log(`Pong! ${obj.data.message}`);
      return;
    }
    else if (obj.type === 'progress') {
      if (!requireBodyField(obj, 'id')) return;
      if (!requireBodyField(obj, 'completeness')) return;

      console.log(`${obj.data.id}: ${obj.data.completeness*100}%`);

      return;
    }
    else if (obj.type === 'inference-result') {
      if (!requireBodyField(obj, 'label')) return;

      console.log(`Inferred ${obj.data.inference}`);

      return;
    }

    sendError('no-such-packet');
  });
};
