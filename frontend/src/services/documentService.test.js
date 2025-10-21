import documentService from './documentService';

describe('DocumentService', () => {
  describe('formatFileSize', () => {
    it('should format bytes correctly', () => {
      expect(documentService.formatFileSize(0)).toBe('0 Bytes');
      expect(documentService.formatFileSize(1024)).toBe('1 KB');
      expect(documentService.formatFileSize(1048576)).toBe('1 MB');
      expect(documentService.formatFileSize(1073741824)).toBe('1 GB');
    });

    it('should handle decimal values', () => {
      expect(documentService.formatFileSize(1536)).toBe('1.5 KB');
      expect(documentService.formatFileSize(1572864)).toBe('1.5 MB');
    });
  });

  describe('formatDate', () => {
    it('should format ISO date string correctly', () => {
      const isoString = '2024-01-15T10:30:00Z';
      const formatted = documentService.formatDate(isoString);
      expect(formatted).toBeTruthy();
      expect(typeof formatted).toBe('string');
    });
  });
});

