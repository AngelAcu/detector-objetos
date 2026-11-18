import { TestBed } from '@angular/core/testing';
import { Detector } from './detector';

describe('Detector', () => {
  let service: Detector;

  beforeEach(() => {
    TestBed.configureTestingModule({});
    service = TestBed.inject(Detector);
  });

  it('should be created', () => {
    expect(service).toBeTruthy();
  });
});
