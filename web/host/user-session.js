import fs from 'fs';
import util from 'util';
import path from 'path';
import child_process from 'child_process';

import del from 'del';
import jwt from 'jsonwebtoken';

const sign = util.promisify(jwt.sign);
const verify = (payload, key) => {
  return new Promise((resolve, reject) => {
    jwt.verify(payload, key, (err, res) => {
      if (err != null) {
        reject(err);
        return;
      }

      resolve(res);
    });
  });
};

const privateKey = fs.readFileSync('rsa/private.pem');
const publicKey = fs.readFileSync('rsa/public.pem');

let lastClientId = 0;

const sessionStore = new Map();

class Session {
  constructor() {
    this.id = lastClientId++;

    this.path = path.join('./uploads/', 'sess-'+this.id);
    this.workingSetPath = path.join(this.path, 'working_set');
    this.labeledPathPrefix = path.join(this.path, 'labeled');

    sessionStore.set(this.id, this);
  }

  classPath(label) {
    return path.join(this.labeledPathPrefix, label);
  }

  filePathFor(name) {
    return path.join(this.workingSetPath, name);
  }

  async jwtpayload() {
    if (this.jwt == null)
      this.jwt = await sign({id: this.id}, privateKey, {algorithm: 'RS256'});

    return this.jwt;
  }

  async setWorkingsetClassName(label) {
    await fs.promises.rename(this.workingSetPath, this.classPath(label));
  }

  async startTraining() {
    this.cp = await child_process.spawn(
      'python3',
      [
        'model.py',
        path.join('..', this.labeledPathPrefix),
        'train',
        this.id,
        'nd'
      ],
      {
        cwd: './ml'
      }
    );
    this.cp.stderr.pipe(process.stderr);
    this.cp.stdout.pipe(process.stdout);
    this.cp.on('exit', (code) => {
      console.log('DONE '+code);
    });
  }

  async removeClass(label) {
    await del(this.classPath(label));
  }

  async cleanup() {
    const actions = [
      del(path.join('./ml', 'class_names', this.id+'_classes.json')),
      del(path.join('./ml', 'models', this.id+'_params.pt')),
      del(path.join('./uploads', 'sess-'+this.id))
    ];

    await Promise.all(actions);
  }
}

export const getSession = async (jwtpayload) => {
  try {
    const decoded = await verify(jwtpayload, publicKey);
    return sessionStore.get(decoded.id);
  }
  catch (e) {
    return null;
  }
};

export const newSession = () => {
  return new Session();
};
