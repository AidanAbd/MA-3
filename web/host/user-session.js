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

  setWS(ws) {
    this.ws = ws;
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
    const proc = child_process.spawn(
      'python3',
      [
        'model.py',
        'train',
        path.join('..', this.labeledPathPrefix),
        this.id,
      ],
      {
        cwd: './ml'
      }
    );
    proc.stdout.on('data', (d) => {
      const args = d.toString().trim().split(' ');

      if (args[0] === 'epoch') {
        this.ws.sendProgress( parseInt(args[1])/parseInt(args[2]) );
        return;
      }
      else if (args[0] === 'done') {
        this.ws.sendProgress(1);
        return;
      }

      console.log(`Got weird response from model.py: "${args.join(' ')}"`);
    });

    await del(this.workingSetPath);
  }

  async removeClass(label) {
    await del(this.classPath(label));
  }

  async cleanup() {
    this.cp.kill();

    const actions = [
      del(path.join('./ml', 'class_names', this.id+'_classes.json')),
      del(path.join('./ml', 'models', this.id+'_params.pt')),
      del(path.join('./uploads', 'sess-'+this.id))
    ];

    await Promise.all(actions);
  }

  inferFile(filename) {
    if (this.cp == null) {
      this.cp = child_process.spawn(
        'python3',
        [
          'model.py',
          'infer',
          this.id
        ],
        {
          cwd: './ml'
        }
      );

      this.cp.on('error', (err) => {
        console.error('Failed to start subprocess. '+err);
      });

      this.cp.stderr.pipe(process.stderr);
      this.cp.stdout.on('data', (d) => {
        const args = d.toString().trim().split(' ');

        if (args[0] === 'debug') {
          console.log(args.join(' '));
          return;
        }
        else if (args[0] === 'res') {
          this.ws.sendInferenceResult(path.basename(args[1]), args[2], args[3]);
          return;
        }

        console.log(`Got weird response from model.py: "${args.join(' ')}"`);
      });
    }
    this.cp.stdin.write(path.join('..', this.filePathFor(filename))+'\n');
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
