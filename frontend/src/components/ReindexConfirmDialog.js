import React, { useState, useEffect } from 'react';
import {
  Dialog,
  DialogTitle,
  DialogContent,
  DialogContentText,
  DialogActions,
  Button,
  FormControlLabel,
  Switch
} from '@mui/material';

function ReindexConfirmDialog({ open, onClose, onConfirm, showAllUsersOption, allUsersSelected }) {
  const [includeAllUsers, setIncludeAllUsers] = useState(allUsersSelected);
  
  // При открытии диалога устанавливаем начальное состояние
  useEffect(() => {
    if (open) {
      setIncludeAllUsers(allUsersSelected);
    }
  }, [open, allUsersSelected]);

  const handleConfirm = () => {
    onConfirm(includeAllUsers);
    onClose();
  };

  return (
    <Dialog
      open={open}
      onClose={onClose}
      aria-labelledby="reindex-dialog-title"
      aria-describedby="reindex-dialog-description"
    >
      <DialogTitle id="reindex-dialog-title">Подтверждение переиндексации</DialogTitle>
      <DialogContent>
        <DialogContentText id="reindex-dialog-description">
          Вы действительно хотите запустить переиндексацию документов? 
          Это может занять некоторое время в зависимости от количества документов.
        </DialogContentText>
        
        {showAllUsersOption && (
          <FormControlLabel
            control={
              <Switch
                checked={includeAllUsers}
                onChange={(e) => setIncludeAllUsers(e.target.checked)}
                color="primary"
              />
            }
            label="Переиндексировать документы всех пользователей"
            sx={{ mt: 2, display: 'block' }}
          />
        )}
      </DialogContent>
      <DialogActions>
        <Button onClick={onClose} color="primary">
          Отмена
        </Button>
        <Button onClick={handleConfirm} color="secondary" variant="contained">
          Переиндексировать
        </Button>
      </DialogActions>
    </Dialog>
  );
}

export default ReindexConfirmDialog; 